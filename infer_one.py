import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration 

class KoBART_title():
    def __init__(self, ckpt_path="./n_title_epoch_3"):
        self.model = BartForConditionalGeneration.from_pretrained(ckpt_path).cuda()
        self.tokenizer = get_kobart_tokenizer()
    def infer(self, text):
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0).cuda()
        output = self.model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

if __name__ == "__main__":
    num = 0
    title_class = KoBART_title()
    while(1):
        num += 1
        c = input(f'{num}: context> ').strip()
        t = title_class.infer(c)
        print(f"Title: {t}")
