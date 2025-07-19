import streamlit as st
try:
    from pa1_new import (
        generate_sentence,
        generate_trigram_sentence,
        generate_fourgram_sentence,
        generate_fivegram_sentence
    )
except ImportError as e:
    st.error(f"Error importing from pa1_new.py: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="N-Gram Text Generator",
    layout="centered"
)

st.title(" N-Gram Text Generator")
st.markdown("---")

# Input section
start_text = st.text_input(
    "Enter starting text:", 
    value="her"
)

n_value = st.selectbox(
    "Choose N-gram level:", 
    options=[2, 3, 4, 5],
    index=0
)

length = st.slider(
    "Sentence length (words):", 
    min_value=5, 
    max_value=30, 
    value=10
)

st.markdown("---")

# Generate button
if st.button("Generate Text", type="primary", use_container_width=True):
    if not start_text.strip():
        st.warning("Please enter some text!")
    else:
        try:
            with st.spinner(f"Generating text using {n_value}-gram model..."):
                if n_value == 2:
                    result = generate_sentence(start_text, length)
                elif n_value == 3:
                    result = generate_trigram_sentence(start_text, length)
                elif n_value == 4:
                    result = generate_fourgram_sentence(start_text, length)
                elif n_value == 5:
                    result = generate_fivegram_sentence(start_text, length)
                else:
                    result = "Unsupported N-gram level."
            
            # Display generated text in a styled box
            if result and result != "Unsupported N-gram level.":
                st.subheader("Generated Text:")
                st.markdown(
                    f"<div style='background-color:#1e1e1e; padding:1em; border-radius:10px; color:white; font-size:16px'>{result}</div>", 
                    unsafe_allow_html=True
                )
        
        except Exception as e:
            st.error(f"Error generating text: {str(e)}")
            st.error("Please check the pa1_new.py file")
