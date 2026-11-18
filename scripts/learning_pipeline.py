"""Small, traced runs of the actual NeuroLM modules; not a benchmark reproduction."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT/'data/tokenizer-cache'))
import numpy as np
import torch
import tiktoken
from torch.utils.data import DataLoader, Subset
from dataset import PickleLoader
from downstream_dataset import WorkloadLoader, HMCLoader
from model.model import GPTConfig
from model.model_neurolm import NeuroLM
from model.model_neural_transformer import NTConfig
from model.model_vq import VQ_Align

RUN = ROOT/'data/runs/learning'
RUN.mkdir(parents=True, exist_ok=True)
ENC = tiktoken.get_encoding('gpt2')


def dump(name, data):
    (RUN/(name+'.json')).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def describe(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        return {'shape':list(x.shape), 'dtype':str(x.dtype),
                'first_values':x.flatten()[:8].float().cpu().tolist()}
    if isinstance(x,(tuple,list)):
        return [describe(t) for t in x]
    return str(x)


def hooks(model, names, trace):
    handles = []
    for name, module in model.named_modules():
        if name in names:
            def hook(mod, inp, out, name=name):
                if name not in trace:
                    trace[name] = {'input':describe(inp), 'output':describe(out)}
            handles.append(module.register_forward_hook(hook))
    return handles


def batch_to_gpu(batch):
    return [t.cuda() for t in batch]


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def load_neurolm(checkpoint):
    conf = GPTConfig(**checkpoint['model_args'])
    model = NeuroLM(conf, init_from='scratch')
    state = {k.removeprefix('_orig_mod.'):v for k,v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    return model.cuda()


def dataset(task, split, val=False):
    folder = ROOT/'data/processed'/('EEGWorkload' if task=='Workload' else 'HMC')/split
    cls = WorkloadLoader if task=='Workload' else HMCLoader
    return cls(str(folder), sorted(p.name for p in folder.glob('*.pkl')),
               is_instruct=True, is_val=val, eeg_max_len=276, text_max_len=80)


def evaluate(model, task, split):
    ds = dataset(task, split, True)
    # Fixed first two examples per class, chosen without looking at predictions.
    selected, counts = [], {}
    for i, name in enumerate(ds.files):
        with (Path(ds.root)/name).open('rb') as f:
            label = int(pickle.load(f)['y'])
        if counts.get(label,0) < 2:
            selected.append(i); counts[label] = counts.get(label,0)+1
    records = []
    model.eval()
    with torch.no_grad(), autocast():
        for i in selected:
            x, text, y, chans, time, valid, mask = ds[i]
            inputs = [t.unsqueeze(0).cuda() for t in (x,text,chans,time,valid,mask)]
            output = model.generate(*inputs[:5], eeg_text_mask=inputs[5], max_new_tokens=5, top_k=1)
            answer = ENC.decode(output[0,text.numel():].tolist())
            if task == 'Workload':
                pred = 1 if answer.strip().startswith('Yes') else 0 if answer.strip().startswith('No') else -1
            else:
                first = answer.strip()[:1]
                pred = 'ABCDE'.index(first) if first and first in 'ABCDE' else -1
            records.append({'file':ds.files[i], 'true_label':y,
                            'prompt':ENC.decode(text[1:].tolist()),
                            'generated_token_ids':output[0,text.numel():].tolist(),
                            'generated_answer':answer, 'predicted_label':pred})
    return {'scope':'two fixed windows per class; demonstration only',
            'accuracy':sum(r['predicted_label']==r['true_label'] for r in records)/len(records),
            'invalid_outputs':sum(r['predicted_label']==-1 for r in records), 'predictions':records}


def instruction(checkpoint, stage, tasks, steps):
    model = load_neurolm(checkpoint)
    trace = {}
    hs = hooks(model, {'tokenizer.patch_embed','tokenizer.blocks.0','tokenizer',
                       'encode_transform_layer','GPT2.transformer.h.0','GPT2.lm_head'}, trace)
    datasets = {t:dataset(t,'train') for t in tasks}
    orders = {t:torch.randperm(len(datasets[t])).tolist() for t in tasks}
    loaders = {t:iter(DataLoader(Subset(datasets[t],orders[t]),batch_size=2,num_workers=0)) for t in tasks}
    cursors = {t:0 for t in tasks}
    trace['task_inputs'] = {}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    losses = []
    for step in range(steps):
        task = tasks[step%len(tasks)]
        batch = batch_to_gpu(next(loaders[task]))
        x,text,y,chans,time,valid,mask = batch
        indices = orders[task][cursors[task]:cursors[task]+2]
        files = [datasets[task].files[i] for i in indices]
        cursors[task] += 2
        if task not in trace['task_inputs']:
            sources = []
            for filename in files:
                with (Path(datasets[task].root)/filename).open('rb') as f:
                    sample = pickle.load(f)
                sources.append({'processed_file':filename, 'raw_file':sample['source'],
                                'start_seconds':sample['start_seconds'], 'label':sample['y']})
            trace['task_inputs'][task] = {'sources':sources,'tensors':describe(batch),
                'valid_patch_counts':valid.sum(-1).cpu().tolist(),
                'full_training_text':ENC.decode(text[0,1:].tolist()),
                'supervised_answer':ENC.decode(y[0,y[0]!=-1].tolist())}
            torch.save([t.cpu() for t in batch],RUN/(stage+'-'+task+'-batch.pt'))
        # EEG prefix positions are not language-answer supervision.
        y_eeg = torch.full((x.size(0),x.size(1)), -1-model.GPT2.config.vocab_size, device='cuda',dtype=torch.long)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss, log, logits = model(x,y_eeg,text,y,chans,time,valid,eeg_text_mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append({'step':step,'task':task,'files':files,'loss':float(loss),'gradient_norm':float(grad)})
        if step == 0:
            torch.save([t.cpu() for t in batch],RUN/(stage+'-first-batch.pt'))
            trace['loader'] = dict(zip(['eeg','text','answer_targets','channels','time','valid','attention'],map(describe,batch)))
            trace['decoded_training_text'] = ENC.decode(text[0,1:].tolist())
            trace['supervised_text_positions'] = torch.where(y[0]!=-1)[0].cpu().tolist()
            trace['supervised_target_ids'] = y[0,y[0]!=-1].cpu().tolist()
            trace['supervised_targets_decoded'] = ENC.decode(y[0,y[0]!=-1].tolist())
            trace['logits'] = describe(logits)
        print(stage, losses[-1],flush=True)
    for h in hs:h.remove()
    results = {task:{split:evaluate(model,task,split) for split in ('eval','test')} for task in tasks}
    torch.save({'model':model.cpu().state_dict(),'model_args':checkpoint['model_args'],
                'learning_only':True},RUN/(stage+'.pt'))
    dump(stage, {'initialization':'author NeuroLM-B; same initial checkpoint for both runs',
                 'steps':steps,'batch_size':2,'lr':1e-5,'task_schedule':tasks,
                 'losses':losses,'trace':trace,'evaluation':results})
    del model,optimizer; gc.collect(); torch.cuda.empty_cache()


def pretrain(checkpoint, steps):
    # Actual architecture, random VQ initialization, language embeddings reused
    # from the author checkpoint. Small explicit text replaces OpenWebText here.
    state = {k.removeprefix('_orig_mod.'):v for k,v in checkpoint['model'].items()}
    encoder_args = dict(n_layer=12,n_head=12,n_embd=768,block_size=1024,bias=False,
                        dropout=0.,num_classes=0,in_chans=1,out_chans=16)
    decoder_args = dict(n_layer=4,n_head=12,n_embd=768,block_size=1024,bias=False,
                        dropout=0.,num_classes=0,in_chans=128)
    aligned = VQ_Align(NTConfig(**encoder_args),NTConfig(**decoder_args),
                       text_embedding_weight=state['GPT2.transformer.wte.weight'][:50257]).cuda()
    del state
    train_files = sorted((ROOT/'data/processed/MotorImagery/train').glob('*.pkl'))
    data = PickleLoader(train_files)
    text = ('A signal is a sequence of measurements over time. A model learns patterns from training examples. '
            'Validation data are kept separate from training data. ')*16
    ids = torch.tensor(ENC.encode(text)[:129],device='cuda').long().unsqueeze(0)
    dump('text_source',{'source':'explicit educational paragraph in scripts/learning_pipeline.py',
                        'is_openwebtext':False,'text':text,'token_ids':ids[0].tolist()})
    trace = {}
    hs = hooks(aligned,{'VQ.encoder.patch_embed','VQ.encoder','VQ.encode_task_layer',
                         'VQ.quantize','VQ.decoder_freq','VQ.decoder_raw','domain_classifier'},trace)
    opt = torch.optim.AdamW([p for p in aligned.parameters() if p.requires_grad],lr=5e-5)
    losses = []
    for step in range(steps):
        batch = batch_to_gpu([t.unsqueeze(0) for t in data[step%len(data)]])
        x,yfreq,yraw,chans,time,valid = batch
        aligned.train(); opt.zero_grad(set_to_none=True)
        with autocast():
            rec, domain_eeg, log = aligned(x,yfreq,yraw,chans,time,valid,alpha=step/max(steps-1,1))
            domain_text = aligned(ids[:,:128])
            loss = rec+domain_eeg+domain_text
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(aligned.parameters(),1.); opt.step()
        losses.append({'step':step,'total_loss':float(loss),'EEG_domain_loss':float(domain_eeg),
                       'text_domain_loss':float(domain_text),**{k:float(v) for k,v in log.items()}})
        if step==0:
            trace['loader'] = describe(batch)
        print('tokenizer',losses[-1],flush=True)
    for h in hs:h.remove()
    aligned.eval()
    torch.save({'model':aligned.cpu().state_dict(),'encoder_args':encoder_args,
                'decoder_args':decoder_args,'learning_only':True},RUN/'tokenizer.pt')
    dump('tokenizer',{'steps':steps,'trace':trace,'losses':losses,
                       'initialization':'random EEG tokenizer; author text embeddings frozen'})
    # Release the alignment classifier and optimizer before the next stage.
    del opt; gc.collect(); torch.cuda.empty_cache()
    tokenizer = aligned.VQ.cuda().eval()
    for p in tokenizer.parameters():p.requires_grad=False
    model = load_neurolm(checkpoint)
    model.tokenizer.load_state_dict(tokenizer.encoder.state_dict())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    pre_data = PickleLoader(train_files,GPT_training=True)
    trace = {}; losses = []
    hs = hooks(model,{'tokenizer','GPT2.transformer.h.0','GPT2.lm_head'},trace)
    for step in range(steps):
        x,chans,time,valid,mask,nchan,ntoken = pre_data[step%len(pre_data)]
        x,chans,time,valid,mask = batch_to_gpu([t.unsqueeze(0) for t in (x,chans,time,valid,mask)])
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.no_grad(),autocast():
            codes = tokenizer.get_codebook_indices(x,chans,time,valid)
        target = torch.full(codes.shape,-1-model.GPT2.config.vocab_size,device='cuda',dtype=torch.long)
        target[:,:ntoken-nchan] = codes[:,nchan:ntoken]
        with autocast():
            eeg_loss,_,logits = model(x,target,None,None,chans,time,valid,eeg_mask=mask)
            text_loss,_,_ = model(x_text=ids[:,:128],y_text=ids[:,1:129])
            loss = eeg_loss+text_loss
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        losses.append({'step':step,'EEG_loss':float(eeg_loss),'text_loss':float(text_loss)})
        if step==0:
            trace.update({'loader':describe((x,chans,time,valid,mask)),
                          'code_ids_first_second':codes[0,:nchan].cpu().tolist(),
                          'code_ids_next_second':codes[0,nchan:2*nchan].cpu().tolist(),
                          'target_first_second':target[0,:nchan].cpu().tolist(),
                          'ignored_target_value_before_offset':-1-model.GPT2.config.vocab_size,
                          'supervised_patch_count':ntoken-nchan,'logits':describe(logits)})
        print('autoregression',losses[-1],flush=True)
    for h in hs:h.remove()
    # Inspect genuinely held-out S002 input without updating weights.
    val_file = sorted((ROOT/'data/processed/MotorImagery/val').glob('*.pkl'))[0]
    model.eval()
    vx,vc,vt,vm,vg,vnc,vnt = PickleLoader([val_file],GPT_training=True)[0]
    vx,vc,vt,vm,vg = batch_to_gpu([t.unsqueeze(0) for t in (vx,vc,vt,vm,vg)])
    with torch.no_grad(),autocast():
        vcode = tokenizer.get_codebook_indices(vx,vc,vt,vm)
        vy = torch.full(vcode.shape,-1-model.GPT2.config.vocab_size,device='cuda',dtype=torch.long)
        vy[:,:vnt-vnc] = vcode[:,vnc:vnt]
        val_loss,_,vlogits = model(vx,vy,None,None,vc,vt,vm,eeg_mask=vg)
    dump('autoregression',{'steps':steps,'losses':losses,'trace':trace,
         'heldout':{'file':val_file.name,'EEG_loss':float(val_loss),
                    'predicted_global_ids_first_8':vlogits[0,:8].argmax(-1).cpu().tolist(),
                    'true_global_ids_first_8':(vy[0,:8]+50304).cpu().tolist()}})
    torch.save({'model':model.cpu().state_dict(),'model_args':checkpoint['model_args'],
                'learning_only':True},RUN/'autoregression.pt')


def inspect_tokenizer():
    ckpt = torch.load(RUN/'tokenizer.pt',map_location='cpu',weights_only=False)
    aligned = VQ_Align(NTConfig(**ckpt['encoder_args']),NTConfig(**ckpt['decoder_args']),
                       text_embedding_weight=ckpt['model']['wte.weight'])
    aligned.load_state_dict(ckpt['model'],strict=True)
    tokenizer = aligned.VQ.cuda().eval()
    path = sorted((ROOT/'data/processed/MotorImagery/val').glob('*.pkl'))[0]
    x,yfreq,yraw,chans,time,valid = batch_to_gpu([t.unsqueeze(0) for t in PickleLoader([path])[0]])
    mask = valid.unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
    with torch.no_grad(),autocast():
        quant,indices,qloss,features = tokenizer.encode(x,chans,time,mask)
        freq,raw = tokenizer.decode(quant,chans,time,mask)
    dump('tokenizer_inspection',{
        'checkpoint':'tokenizer.pt','file':path.name,'slurm_job':os.environ.get('SLURM_JOB_ID'),
        'updates':0,'input':describe(x),'encoder_features':describe(features),
        'quantized_vectors':describe(quant),'code_ids_first_8':indices[:8].cpu().tolist(),
        'frequency_target':describe(yfreq),'frequency_reconstruction':describe(freq),
        'waveform_target':describe(yraw),'waveform_reconstruction':describe(raw),
        'frequency_mse':float(torch.nn.functional.mse_loss(freq.float(),yfreq)),
        'waveform_mse':float(torch.nn.functional.mse_loss(raw.float(),yraw)),
        'quantization_loss':float(qloss)})
    print('TOKENIZER HELDOUT RECONSTRUCTION INSPECTION COMPLETE',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--inspect-tokenizer',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    if args.inspect_tokenizer:
        inspect_tokenizer()
        return
    checkpoint = torch.load(ROOT/'data/models/NeuroLM-B.pt',map_location='cpu',weights_only=False)
    dump('environment',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        'slurm_job':os.environ.get('SLURM_JOB_ID'),'seed':42,'steps':args.steps,
                        'checkpoint_keys':list(checkpoint), 'model_args':checkpoint['model_args']})
    instruction(checkpoint,'workload',['Workload'],args.steps)
    instruction(checkpoint,'multitask',['Workload','HMC'],args.steps)
    pretrain(checkpoint,args.steps)
    print('ALL FOUR LEARNING STAGES COMPLETE',flush=True)


if __name__=='__main__':main()
