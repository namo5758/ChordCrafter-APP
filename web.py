import streamlit as st
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
import torch
import chord_audi as ca 

torch.classes.__path__ = []     

MODEL_DIR = "namo5758/ChordCrafter-Model"
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def main():
    st.title("ChordCrafter")
    st.subheader("Generate Chord Progressions with Sound")

    history = st.text_input("Enter your starting chords (2-4 chords):")
    if history:
        ids     = tokenizer.encode(history + " <bos>", return_tensors="pt").to(device)
        gen_ids = model.generate(
            ids, max_new_tokens=32, do_sample=True, num_beams=1,
            top_p=0.9, temperature=1.1, no_repeat_ngram_size=3,
            repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id
        )
        full_pred   = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred_chords = full_pred.split()
        new_chords  = pred_chords[len(history.split()):]

        st.markdown("**Suggested progression:**")
        st.code(" ".join(new_chords))

        if st.button("Play progression"):
            audio_bytes, skipped, errors = ca.generate_audio(
                    new_chords,
                    program=0,  # piano
                    include_metronome=False
                )
            
            st.write(f"Generated WAV size: {len(audio_bytes):,} bytes")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.error("Audio buffer is empty – nothing to play.")

if __name__ == "__main__":
    main()