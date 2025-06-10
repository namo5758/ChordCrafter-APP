import streamlit as st
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
import torch

torch.classes.__path__ = []

model_dir = "namo5758/ChordCrafter-Model"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
model     = AutoModelForCausalLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def main():
    st.title("ChordCrafter")
    st.subheader("Generate Chord Progressions")
    
    history = st.text_input("Enter your chords (2 or 4 chords):")
    if history:
        input_ids = tokenizer.encode(history + " <bos>", return_tensors="pt").to(device)
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=True,
            num_beams=1,
            top_p=0.9,
            temperature=1.1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        full_pred  = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred_chords = full_pred.split()
        new_chords = pred_chords[len(history.split()):]
        st.write(" ".join(new_chords))

if __name__ == "__main__":
    main()