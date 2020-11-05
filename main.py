# %%
import torch
import string

# from transformers import BertTokenizer, BertForMaskedLM
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig

unifiedqa_t5_large = "allenai/unifiedqa-t5-large" # you can specify the model size here
unifiedqa_t5_large_tok = AutoTokenizer.from_pretrained(unifiedqa_t5_large)
unifiedqa_t5_large = T5ForConditionalGeneration.from_pretrained(unifiedqa_t5_large).eval()

unifiedqa_t5_3B = "allenai/unifiedqa-t5-3b" # you can specify the model size here
TBconfig = AutoConfig.from_pretrained('t5-3b')
unifiedqa_t5_3B_tok = AutoTokenizer.from_pretrained(unifiedqa_t5_3B)
# Hot fix until huggingface issue is fixed
unifiedqa_t5_3B = T5ForConditionalGeneration.from_pretrained("./unifiedqa/3B/model.ckpt-1100500.index", from_tf=True, config=TBconfig).eval()

# unifiedqa_t5_large = "allenai/unifiedqa-t5-large" # you can specify the model size here
largeconfig = AutoConfig.from_pretrained('t5-large')
t5_large_tok = AutoTokenizer.from_pretrained("t5-large")
t5_large = T5ForConditionalGeneration.from_pretrained("./t5_large/model.ckpt-1025700.index", from_tf=True, config=largeconfig).eval()

t5_3B_tok = AutoTokenizer.from_pretrained("t5-3b")
t5_3B = T5ForConditionalGeneration.from_pretrained("./t5_3b/model.ckpt-1025500.index", from_tf=True, config=TBconfig).eval()

# from transformers import XLNetTokenizer, XLNetLMHeadModel
# xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large').eval()

# from transformers import BartTokenizer, BartForConditionalGeneration
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

# from transformers import ElectraTokenizer, ElectraForMaskedLM
# electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
# electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-large').eval()

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def run_t5(input_string, model, tokenizer, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return [tokenizer.decode(x) for x in res]

def run_pt_t5(input_string, model, tokenizer,choices, **generator_args):
    tokenized = tokenizer.batch_encode_plus(
        [input_string], max_length=512, return_tensors="pt",
    )

    generated_ids = model.generate(input_ids=tokenized['input_ids'], **generator_args)
    preds = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]
    # print(preds)
    return num_to_choice(preds, choices)

def num_to_choice(answers, choices):
    for idx, answer in enumerate(answers):
        if len(answer.strip()) < 2:
            try:
                answer = int(answer)
                if answer<len(choices):
                    answers[idx] = choices[answer]
            except:
                pass
    return answers



def get_all_predictions(text_sentence, question="", choices="", top_clean=5):
    # ========================= Unified QA ================================
    uqa_input = question
    choices = choices.split('\n')
    joined = []
    for i, choice in enumerate(choices):
        joined.append(f"({chr(ord('a')+i)}) {choice}")
    joined = " ".join(joined)
    uqa_input = f"{question} \\n {joined} \\n {text_sentence}"
    print(f"uqa input: {uqa_input}")
    uqa_large = '\n'.join(run_t5(uqa_input, unifiedqa_t5_large, unifiedqa_t5_large_tok, num_beams=4*top_clean, num_return_sequences=top_clean))
    # print(uqa_large)

    uqa_3b = '\n'.join(run_t5(uqa_input, unifiedqa_t5_3B, unifiedqa_t5_3B_tok, num_beams=4*top_clean, num_return_sequences=top_clean))


    joined_t5 = []
    for i, choice in enumerate(choices):
        joined_t5.append(f"choice {i}: <{choice}>")
    joined_t5 = " ".join(joined_t5)

    t5_inp = f"question: <{question}> {joined_t5} article {text_sentence}"

    print(f"t5 input: {t5_inp}")
    t5_large_op = '\n'.join(run_pt_t5("trivia question: " + t5_inp.lower(), t5_large, t5_large_tok, choices,  num_beams=4*top_clean, num_return_sequences=top_clean))

    t5_3B_op = '\n'.join(run_pt_t5(t5_inp.lower(), t5_3B, t5_3B_tok, choices, num_beams=4*top_clean, num_return_sequences=top_clean))


    # ========================= BERT =================================
    # print(text_sentence)
    # input_ids, mask_idx = encode(bert_tokenizer, text_sentence+ " <mask>")
    # with torch.no_grad():
    #     predict = bert_model(input_ids)[0]
    # bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= XLNET LARGE =================================
    # input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    # perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    # perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    # target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    # target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    # with torch.no_grad():
    #     predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    # xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= XLM ROBERTA BASE =================================
    print(f"Rest input: Prompt: {text_sentence} question: {question} {joined_t5}+ The correct choice is choice number: <mask>")
    input_ids, mask_idx = encode(xlmroberta_tokenizer, f"Prompt: {text_sentence} question: {question} {joined_t5}+ The correct choice is choice number: <mask>", add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= BART =================================
    # input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = bart_model(input_ids)[0]
    # bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= ELECTRA =================================
    # input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = electra_model(input_ids)[0]
    # electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= ROBERTA =================================
    input_ids, mask_idx = encode(roberta_tokenizer, f"Prompt: {text_sentence} question: {question} {joined_t5}+ The correct choice is choice number: <mask>", add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    opdct = {'uqalarge': uqa_large,
            'uqa3B' : uqa_3b,
            't5_large': t5_large_op,
            't5_3b': t5_3B_op,
            # 'bert': bert,
            # 'xlnet': bert,
            'xlm': xlm,
            # 'bart': bert,
            # 'electra': bert,
            'roberta': roberta}

    print("Outputs:")
    for k in opdct:
        print(f"{k} : {opdct[k]}")

    return opdct

    # return {'bert': bert,
    #         'xlnet': xlnet,
    #         'xlm': xlm,
    #         'bart': bart,
    #         'electra': electra,
    #         'roberta': roberta}
