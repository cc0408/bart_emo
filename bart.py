import os
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, BartForConditionalGeneration, BartForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader, Dataset
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="try.")
    
    parser.add_argument("--model", default="facebook/bart-base", type=str,
        help="type of model")
    parser.add_argument("--lr", default=3e-1, type=float,
        help="learning rate")
    parser.add_argument("--embed_layer", default=-1, type=int,
        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,
        help="embedding similarity regularizer")
    parser.add_argument("--gpt2_checkpoint_folder", default="result/", type=str,
        help="folder for loading GPT2 model trained with BERT tokenizer")
    parser.add_argument("--batch_size", default=8, type=int,
        help="batch size for gumbel-softmax samples")
    parser.add_argument("--kappa", default=5, type=float,
        help="CW loss margin")
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder for loading trained models")
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2", "sst5"],
        help="classification dataset to use")
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).to(device)
    ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))
    victim_model = AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2).cuda()
    embeddings = victim_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())

    suffix = '_finetune'
    model_checkpoint = os.path.join(args.result_folder, '%s_%s%s.pth' % (args.model.replace('/', '-'), args.dataset, suffix))
    print('Loading checkpoint: %s' % model_checkpoint)
    model.load_state_dict(torch.load(model_checkpoint))

    raw_dataset = load_dataset("glue", 'sst2')
    #preprocess_function = lambda examples: tokenizer(examples['sentence'], examples['label'], max_length=256, truncation=True)
    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence','idx'])
    tokenized_datasets.set_format('torch')
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)  # 通过这里的dataloader，每个batch的seq_len可能不同
    eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=args.batch_size, collate_fn=data_collator)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #model.train()
    for idx,batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        #break
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        orig_output = ref_model(input_ids=torch.LongTensor(batch['input_ids'].cpu()).to(device), 
                                attention_mask=torch.LongTensor(batch['attention_mask'].cpu()).to(device)).hidden_states[args.embed_layer]
        ref_embeds = (outputs.logits @ ref_embeddings[None, :, :])
        pred = ref_model(inputs_embeds=ref_embeds)
        output = pred.hidden_states[args.embed_layer]
        cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
        ref_loss = -args.lam_sim * cosine.mean()

        inputs_embeds = (outputs.logits @ embeddings[None, :, :])
        pred = victim_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=inputs_embeds).logits
        top_preds = pred.sort(descending=True)[1]
        label = batch['labels']
        correct = (top_preds[:, 0] == label).long()
        indices = top_preds.gather(1, correct.view(-1, 1))
        adv_loss = (pred[:, label] - pred.gather(1, indices).squeeze() + args.kappa).clamp(min=0).mean()
        
        loss = ref_loss + adv_loss
        if idx % 100 ==0:
            print(loss)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
            
        optimizer.step()
        
    #print(torch.LongTensor(batch['input_ids']))
