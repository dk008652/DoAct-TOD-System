#!/usr/bin/env python
# coding: utf-8
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, PreTrainedModel, GPT2Config
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader  #line edited
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import global_config as cfg 
# from config21 import global_config as cfg  # global, already initialized

import warnings
warnings.filterwarnings("ignore")

domain_act_label_map = {'restaurant_inform':0,'restaurant_request':1, 'restaurant_nooffer':2,'restaurant_recommend':3, 'restaurant_select':4,'restaurant_offerbook':5, 'restaurant_offerbooked':6,'restaurant_nobook':7, 'restaurant_reqmore':8,
                        'hotel_inform':9,'hotel_request':10, 'hotel_nooffer':11,'hotel_recommend':12, 'hotel_select':13,'hotel_offerbook':14, 'hotel_offerbooked':15,'hotel_nobook':16, 'hotel_reqmore':17, 
                        'train_inform':18,'train_request':19, 'train_nooffer':20,'train_recommend':21, 'train_select':22,'train_offerbook':23, 'train_offerbooked':24,'train_nobook':25, 'train_reqmore':26, 
                        'attraction_inform':27,'attraction_request':28, 'attraction_nooffer':29,'attraction_recommend':30, 'attraction_select':31,'attraction_offerbook':32, 'attraction_offerbooked':33,'attraction_nobook':34, 'attraction_reqmore':35, 
                        'taxi_inform':36,'taxi_request':37, 'taxi_nooffer':38,'taxi_recommend':39, 'taxi_select':40,'taxi_offerbook':41, 'taxi_offerbooked':42,'taxi_nobook':43, 'taxi_reqmore':44, 
                        'hospital_inform':45,'hospital_request':46, 'hospital_nooffer':47,'hospital_recommend':48, 'hospital_select':49,'hospital_offerbook':50, 'hospital_offerbooked':51,'hospital_nobook':52, 'hospital_reqmore':53, 
                        'police_inform':54,'police_request':55, 'police_nooffer':56,'police_recommend':57, 'police_select':58,'police_offerbook':59, 'police_offerbooked':60,'police_nobook':61, 'police_reqmore':62,} 
                       
domains = ['restaurant', 'hotel', 'train', 'attraction', 'taxi', 'hospital', 'police']  # Add your 7 domains
acts = ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook', 'reqmore']  # Add your 9 acts

# Lists for domains and acts
domains_act = ['[restaurant]', '[hotel]', '[train]', '[attraction]','[taxi]', '[hospital]' ,'[police]' , '[inform]' ,'[request]', '[nooffer]',
               '[recommend]', '[select]' ,'[offerbook]', '[offerbooked]', '[nobook]' ,'[reqmore]']  # domains and acts are used to average for domain_system_act classificaion



class base_model(PreTrainedModel):
    def __init__(self, vocab_size, classes = 63):
        config = GPT2Config()
        config.output_attentions = True
        super(base_model,self).__init__(config)
        self.gpt = GPT2Model.from_pretrained('distilgpt2', output_attentions=True)
        # self.gpt = GPT2LMHeadModel.from_pretrained('distilgpt2', output_attentions=True)

        self.embed_size = self.gpt.config.n_embd
        # self.linear_embed = nn.Linear(self.embed_size, self.embed_size)
        self.num_domain = classes

        self.logit = nn.Linear(self.embed_size, vocab_size)
        self.linear_c1 = nn.Linear(self.embed_size, self.embed_size//4)
        self.linear_c2 = nn.Linear(self.embed_size//4, self.num_domain)
        # self.copylayer = CopyLayer(self.embed_size)
    
    def forward(self, inputs, domain_act_index=None):
        position_embeddings = self.gpt.wpe.weight  # Word Position Embeddings

        new_token_embed = torch.Tensor()
        for i in range(inputs.size(0)):  #loop over the batch size
            #token_embed = word_embeddings[inputs[i, :], :]  #word embedding of i-th batch
            token_embed = self.gpt.get_input_embeddings()(inputs[i])                #turn and role embedding
            pos_embed = position_embeddings[:inputs.size(-1),:]
            token_embed  = pos_embed.unsqueeze(0) + token_embed.unsqueeze(0) 
            new_token_embed = torch.cat((new_token_embed.to(self.device), token_embed.to(self.device)), dim=0)
        # new_token_embed = self.linear_embed (new_token_embed)   #adding linear layer after embedding
        
        out= self.gpt(inputs_embeds=new_token_embed, output_attentions=True)
        lm_logits = self.logit(out[0])
        ## use out[0] for creating masked output containing only domain and system act
        ###########################################################################
        if domain_act_index != None:
            hidden_states_of_interest = []
            for i, indices in enumerate(domain_act_index):
                # Extract the hidden states for the specified indices for each batch input
                hidden_states = out[0][i, indices, :]  # Shape: [len(indices), hidden_size]
                hidden_states_of_interest.append(hidden_states)
            # hidden_states_of_interest = torch.tensor(hidden_states_of_interest)
            hidden_states_of_interest = rnn_utils.pad_sequence(hidden_states_of_interest, batch_first=True)
            out_class = torch.mean(hidden_states_of_interest, dim=1).to(self.device)
            out_class = self.linear_c1(out_class)
            # out_class = nn.functional.sigmoid( self.linear_c2(out_class))
            out_class = nn.functional.softmax( self.linear_c2(out_class), dim=-1)
        
        ###########################################################################
            
            return lm_logits, out_class
        else:
            return lm_logits, None

        # masked_output = out[0]
        # print("shape of masked output", masked_output.shape)
        # out_copy = self.copylayer(inputs, lm_logits, out[0], out[-1], self.gpt.wte)
        
        
        # print(lm_logits.shape)
        # print(type(lm_logits))
        

class Model(object):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        # initialize tokenizer        
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)
        #if cfg.mode == 'train':
        self.vocab_size = len(self.tokenizer)
        if cfg.gpt_path == 'distilgpt2':
            self.model = base_model(self.vocab_size).to(self.device)
        else:
            logging.info('MODEL LOADED FROM THE DIRECTORY')
            # self.model = base_model.from_pretrained(cfg.gpt_path+"/pytorch_model.bin").to(self.device)
            self.model = base_model(self.vocab_size).to(self.device)
            self.model.gpt.resize_token_embeddings(len(self.tokenizer))
            self.model.load_state_dict(torch.load(cfg.gpt_path+"/pytorch_model.bin"))
        # print(self.model)
        # print("len of tokenizer: ", len(self.tokenizer))
        if cfg.mode == 'train':
            #PATH = 'experiments/all__sd11_lr0.0001_bs2_ga16/epoch46_trloss0.74_gpt2/epoch46_trloss0.74_gpt2.bin'
            self.model.gpt.resize_token_embeddings(len(self.tokenizer))
        # self.model.to(self.device)  # single gpu
        self.evaluator = MultiWozEvaluator(self.reader)
        num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Number of parameters: %d", num_params)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log_DSA')
        else:
            self.tb_writer = None
    
    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        #weight decay is termed as a L2 normalization//  confirm it again
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] *            cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler
    
    def domain_act_decode(self):
        self.domain_act_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in domains_act]
        # print("domain_act_token_ids", self.domain_act_token_ids)


    def domain_act_label(self, batch):
        # self.domain_act_decode()
        batch_DSA = []
        for context in batch['contexts']:
            batch_DSA.append(self.tokenizer.decode(context))
        # print("batch_DSA")
        # print(len(batch_DSA)) 
        # print(batch_DSA)

        # Process each component in the sentence
        final_domain_act = []
        for i in range(len(batch_DSA)):
            # components = batch[i].split()
            domain_acts = []
            domain = None

            for component in batch_DSA[i].split():
                if component.startswith('[') and component.endswith(']'):
                    # Strip brackets and check if it's a domain or an act
                    label = component[1:-1]
                    if label in domains:
                        domain = label  # Update the current domain
                    elif label in acts:
                        # If it's an act and we have a domain, append domain_act
                        if domain:
                            domain_acts.append(f'{domain}_{label}')
            final_domain_act.append(domain_acts)
        # Output the result
        
        self.num_domain_act = 63  # Total number of domains_act (9x7 = 63)
        domain_act_label = torch.zeros(len(final_domain_act), self.num_domain_act)  # Initialize label tensor

        for i, dialogue_domains_act in enumerate(final_domain_act):
            for domain_act in dialogue_domains_act:
                domain_act_label[i, domain_act_label_map[domain_act]] = 1
        # print(domain_act_label)

        return domain_act_label


    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:2]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        # it will convert the contexts_np to context_tensor and size of context_tensor will be the maximum of the context_np 
        #example if context_np = [120, 181, 104, 175, 117] than context_tensor will be the size of [181,181,181,181] 
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs
        
    def calculate_loss_and_accuracy(self, outputs, out_c, labels, label_DSA):
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_domain = loss_fn(out_c, label_DSA.float().to(self.device))

        lm_logits = outputs  # len of lm_logits = size of batch {batch_size}
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()
        
        loss /= num_targets

        alpha = cfg.wt
        # print("Domain Loss: ",loss_domain)
        return loss, alpha*loss_domain
        
    def train(self):
        all_batches = self.reader.get_batches('train')
        optimizer, scheduler = self.get_optimizers()
        set_stats = self.reader.set_stats['train']
        cfg.pad_id = self.tokenizer.encode('<pad>')[0]
        #print("pad id:", cfg.pad_id)   
        
        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  weightage to system_act loss cfg.wt %.2f : ", cfg.wt)
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_dials']*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))

        # tb writer
        print("tb_writer: ", self.tb_writer)
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))

        log_inputs = 0
        global_step = 0
        sw = time.time()

        # self.domain_act_decode()
        self.domain_act_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in domains_act]

        for epoch in range(cfg.epoch_num):
            logging.info("******Epoch: {} ********".format(epoch+1))
            # cfg.mode = 'train'
            epoch_step = 0
            tr_loss = 0.0
            dom_loss = 0.0 # new
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            
            data_iterator = self.reader.get_nontranspose_data_iterator(all_batches) #line modified all_batches[0:10]
        
            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    self.model.train()
                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    label_DSA = self.domain_act_label(inputs)
                    mask = self.create_mask(inputs['contexts_tensor'])
                    context_system_act = inputs['contexts_tensor'] * mask
                    domain_act_index = self.create_index_for_domain_act(context_system_act)
                    outputs, out_c = self.model(inputs['contexts_tensor'], domain_act_index)
                    
                    loss, domain_loss = self.calculate_loss_and_accuracy(outputs, out_c, labels=inputs['contexts_tensor'], label_DSA=label_DSA)
                    loss.backward(retain_graph=True)
                    domain_loss.backward()
                    tr_loss += loss.item()
                    dom_loss += domain_loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                        # end of an epoch
                        (epoch_step + \
                         1) == set_stats['num_training_steps_per_epoch']
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) /                                 cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))
                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and epoch+1>=cfg.start_epoch:
                                
                                logging.info("TESTING, Epoch No: {}".format(epoch+1))
                                results_test = self.validate('test')
                                if results_test['score']>106:
                                    logging.info("VALIDATION, Epoch No: {}".format(epoch+1))
                                    results = self.validate()
                                
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format((time.time()-btm)/60, tr_loss))
    
    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        # torch.save(self.model.state_dict(), save_path)
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg
    
    def validate(self,data='dev', do_test=False):
        self.model.eval()
        eval_data = self.reader.get_eval_data(data)
        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            for dial_idx, dialog in (enumerate(eval_data)):
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    
                    inputs = self.add_torch_input_eval(inputs) #input["context_tensor"]

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    
                    if cfg.use_true_curr_bspn: # generate act, response #generaring act and response based on the oracle belief state
                        max_len=60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80
                        
                        if (context_length+60<1024):
                            max_len_ = 60
                        else:
                            max_len_ = int(1024-context_length)
                        
                        outputs = self.custom_generate_beam_search_decoder(inputs['context_tensor'], 1, max_len_, self.tokenizer.encode(['<eos_b>'])[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0]
                        # generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        #print("********with/else part is running********")
                        top_k=5
                        top_p=0.95
                        if (context_length+60<1024):
                            max_len_ = 60
                        else:
                            max_len_ = int(1024-context_length)
                        
                        outputs = self.custom_generate_beam_search_decoder(inputs['context_tensor'], 1, max_len_, self.tokenizer.encode(['<eos_b>'])[0])
                        generated_bs = outputs[0]
                        # generated_bs = list(generated_bs)
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        # check DB result
                        if cfg.use_true_db_pointer:
                            # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            db = turn['db']
                        else:
                            #print("db_result/self.tokenizer.decode(bspn_gen)", self.tokenizer.decode(bspn_gen))
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                            db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
                            
                        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + list(bspn_gen) + db]).to(self.device)
                        context_length = len(inputs['context_tensor_db'][0])
                        
                        if (context_length+80<1024):
                            max_len_ = 80
                        else:
                            max_len_ = int(1024-context_length)
                        
                        outputs_db = self.custom_generate_beam_search_decoder(inputs['context_tensor_db'], 1, max_len_, self.tokenizer.encode(['<eos_r>'])[0])
                        generated_ar = outputs_db[0]
                        generated_ar = generated_ar[context_length-1:]
                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                            
                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']   #what is dspn

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                result_collection.update(self.reader.inverse_transpose_turn(dialog))

        logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)

        bleu, success, match = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
            match, success, bleu, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)
        
        return eval_results
    
    def top_k_top_p(self, logits, top_k, filter_value=-float('Inf')):
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        score = sorted_logits[0][:top_k].tolist()
        sequence = sorted_indices[0][:top_k]
        return sequence, score

    def custom_generate_beam_search_decoder(self, inputs, beam_width, seq_len, eos):
        temperature = 0.7
        logits, _ = self.model(inputs)
        lm_logits = logits #line modified to make it compatible for custom model
        shift_logits = lm_logits[...,-1, :] #this is the logit for last token prediction
        shift_logits.unsqueeze(1)
        sequence, score = self.top_k_top_p(shift_logits, beam_width)
        flag = 0
        beams = []
        for i in range(beam_width):
            beams.append((torch.tensor([sequence[i]]), score[i]))

        for t in range(seq_len):
            new_beams = []
            for beam in (beams):
                sequence, score = beam
                inputs = inputs.view(-1)
                inputs = inputs.to(self.device)
                sequence = sequence.to(self.device)
                logits, _ = self.model(torch.cat((inputs, sequence)).unsqueeze(0)) #edit
                logits = logits / temperature  #line modified to make it compatible for custom model
                shift_logits = logits[...,-1, :]
                sequence_k, score_k = self.top_k_top_p(shift_logits, beam_width)

                if eos in sequence_k: #eos = end of sequence// self.tokenizer.encode([<eos_b>])
                    flag = 1
                    break
                for i in range(beam_width):
                    temp_beam = ( torch.cat((sequence.to(self.device), torch.tensor([sequence_k[i]]).to(self.device) ) ), score+score_k[i] )  #find a way to append tensor but 
                    #it should not change the original sequence
                    new_beams.append(temp_beam)
            new_beams = sorted(new_beams, key = lambda val: val[1], reverse=True)[:beam_width]  #sort based on the score
            if new_beams != []:
                beams = new_beams
            if flag:
                break
        top_sequence = beams[-1][0]   #recheck indexing 
        return torch.cat((inputs.to(self.device), top_sequence.to(self.device))).unsqueeze(0).tolist() ##line modified to make it compatible for custom model
    
    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated)-1
        
        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx+1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx+1]
            decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]

        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated)-1
        return generated[: eos_b_idx+1]

    def create_index_for_domain_act(self, context_system_act):
        self.domain_act_token_ids
        system_act_index_in_batch = []
        
        for i in range(len(context_system_act)):
            system_act_index = torch.nonzero(context_system_act[i].unsqueeze(1) == torch.tensor(self.domain_act_token_ids, device=self.device), as_tuple=False)[:, 0]
            system_act_index_in_batch.append(system_act_index)
        
        return system_act_index_in_batch



    
    def create_mask (self, inputs):
        sos_a_id = self.tokenizer.encode(['<sos_a>'])[0] # segment 4
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        # creating mask for extracting user utterance and response from input for the classification purpose
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for i in range(inputs.size(0)):  #loop over the batch size
            sos_a_idx = (inputs[i] == sos_a_id).nonzero(as_tuple=True)[0]
            eos_a_idx = (inputs[i] == eos_a_id).nonzero(as_tuple=True)[0]
            
            if(len(sos_a_idx)==len(eos_a_idx)):
                j = 0
                while( j<sos_a_idx.size(0) and j<eos_a_idx.size(0) ):
                    element1 = sos_a_idx[j].item()
                    element2 = eos_a_idx[j].item()
                    # create mask
                    mask[i, element1:element2+1] = True
                    j += 1
        return mask

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return

def _init_logging_handler(mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log_DSA'):
            os.mkdir('./log_DSA')
        if cfg.save_log and cfg.mode == 'train':
            file_handler = logging.FileHandler('./log_DSA/log_DSA{}_{}_{}_{}_sd{}.txt'.format(cfg.log_time, mode, '-'.join(cfg.exp_domains), cfg.exp_no, cfg.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif cfg.mode == 'test':
            eval_log_path = os.path.join(cfg.eval_load_path, 'eval_log.json')
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

def main():
    if not os.path.exists('./experiments_DSA'):
        os.mkdir('./experiments_DSA')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if cfg.paraphrased == True:
        data_path_paraphrased()
    
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        # cfg.model_path = cfg.eval_load_path
        cfg.gpt_path = cfg.eval_load_path
        print("self.eval_load_path: ", cfg.eval_load_path)
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            
            experiments_path = './experiments_DSA' if 'all' in cfg.exp_domains else './experiments_Xdomain'
            
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps))
            if not os.path.exists('./experiments_Xdomain'):
                os.mkdir('./experiments_Xdomain')
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)
            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path
    
    _init_logging_handler(args.mode)
    logging.info('save path:'.format(cfg.exp_path))
    if cfg.cuda:
        if len(cfg.cuda_device) == 2:
            cfg.multi_gpu = False
            device = torch.device("cuda:{}".format(cfg.cuda_device[cfg.cuda_id]))
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # initialize model
    m = Model(device)
    if args.mode == 'train':    # train
        if cfg.save_log:  # save cfg details.
            pass
        if cfg.context_scheme == 'UBARU':
            m.train()
        elif cfg.context_scheme == 'URURU':
            m.train_URURU()
        else:
            logging.info('Invalid context Scheme. must be UBARU or URURU')
            exit()
    elif args.mode == 'adjuest':
        pass
    else:  # test
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                        ))

        if cfg.context_scheme == 'UBARU':
            # m.validate()
            m.validate('test')

        elif cfg.context_scheme == 'URURU':
            m.validate_URURU()
            m.validate_URURU('test')

if __name__ == "__main__":
    main()
