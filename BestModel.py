import pandas
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from torch import nn
import string
from huggingface_hub import hf_hub_download
import io
import openpyxl
import xlsxwriter
import os

Username = os.getenv('USERNAME')
Token = os.getenv('TOKEN')

st.set_page_config(
    page_title="MWEs Prediction App",
    page_icon="🚀"
)

# Function to check if a token is punctuation
def is_punctuation(token):
    return all(char in string.punctuation for char in token)

# Define the RoBERTa-CRF Model
class RoBertaCRFModel(nn.Module):
    def __init__(self, model_name, num_labels_mwe):
        super(RoBertaCRFModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mwe = nn.Linear(self.roberta.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(self, input_ids, attention_mask, labels_mwe=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)

        mask = attention_mask.bool()
        if labels_mwe is not None:
            labels_mwe = labels_mwe.clone()
            labels_mwe[labels_mwe == -100] = 0
            loss = -self.crf(logits_mwe, labels_mwe, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits_mwe, mask=mask)
            return predictions

# Load tokenizer and model from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_repo = Username  # Replace with your Hugging Face repo ID
    token = Token  # Replace with your token

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_auth_token=token)

    # Download model weights using Hugging Face Hub
    model_weights_path = hf_hub_download(repo_id=model_repo, filename="roberta_crf_model_weights.pth", use_auth_token=token)

    # Initialize the model
    model = RoBertaCRFModel('roberta-base', num_labels_mwe=len(mwe_label_to_id))

    # Load weights into the model
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer

# Map tag indices to labels
mwe_label_to_id = {'B-MWE': 0, 'I-MWE': 1, 'O': 2}
idx2tag = {v: k for k, v in mwe_label_to_id.items()}

# Function for MWE Detection
def perform_mwe_detection(sentence, model, tokenizer):
    """
    Perform MWE detection using the RoBERTa-CRF model.

    Args:
        sentence (str): Input sentence to analyze.
        model (nn.Module): The loaded RoBERTa-CRF model.
        tokenizer (AutoTokenizer): Tokenizer for the RoBERTa-CRF model.

    Returns:
        table_data (list): Token-level predictions with tags.
        detected_mwes (list): Detected MWEs (multi-word expressions).
    """
    # Tokenize input sentence
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    # Get input tensors
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Make predictions
    with torch.no_grad():
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        crf_predictions = predictions[0]

    # Decode tokens and predictions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    current_mwe_tokens = []
    detected_mwes = []

    table_data = []
    for token, pred_idx in zip(tokens, crf_predictions):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        clean_token = token.lstrip("Ġ")
        pred_tag = idx2tag.get(pred_idx, 'O')

        # If punctuation, force label to "O"
        if is_punctuation(clean_token):
            pred_tag = 'O'

        # Add to table data
        table_data.append({"Token": clean_token, "Prediction": pred_tag})

        # Collect MWEs
        if pred_tag in ['B-MWE', 'I-MWE']:
            current_mwe_tokens.append(clean_token)
        elif current_mwe_tokens:
            mwe_text = ' '.join(current_mwe_tokens)
            detected_mwes.append(mwe_text)
            current_mwe_tokens = []

    # Capture remaining MWE tokens
    if current_mwe_tokens:
        mwe_text = ' '.join(current_mwe_tokens)
        detected_mwes.append(mwe_text)

    return table_data, detected_mwes

# Load model and tokenizer
with st.spinner("Loading the model..."):
    model, tokenizer = load_model_and_tokenizer()

# Sidebar for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Sentence Prediction"

with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #1E90FF;
            margin-bottom: 20px;
        }
        .sidebar-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
        }
        .sidebar-button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">Navigation Bar</div>', unsafe_allow_html=True)
    
    # Manage navigation state
    if st.button("✍️ Sentence Prediction"):
        st.session_state["page"] = "Sentence Prediction"
    if st.button("📄 Excel File Prediction"):
        st.session_state["page"] = "Excel File Prediction"

# Navigate based on session state
if st.session_state["page"] == "Sentence Prediction":
    st.title("✍️ MWEs Detection with RoBERTa-CRF")
    st.write(
        """
        This app detects **Multi-Word Expressions (MWEs)** in a sentence using a fine-tuned 
        **RoBERTa-CRF model**.
        """
    )

    # Add the accuracy and classification report as a table
    st.markdown('<h3 style="color:skyblue;">Model Performance on the Test Set</h3>', unsafe_allow_html=True)
    accuracy = 0.8871
    classification_report = {
        "Metric": ["Precision", "Recall", "F1-Score", "Support"],
        "MWE": [0.62, 0.70, 0.66, 1396],
        "Micro Avg": [0.62, 0.70, 0.66, 1396],
        "Macro Avg": [0.62, 0.70, 0.66, 1396],
        "Weighted Avg": [0.62, 0.70, 0.66, 1396],
    }

    # Create a DataFrame for the classification report
    classification_df = pandas.DataFrame.from_dict(classification_report, orient="index")
    classification_df.columns = classification_df.iloc[0]  # Set the first row as column headers
    classification_df = classification_df[1:]  # Remove the first row

    # Display the accuracy
    st.write(f"**Accuracy**: {accuracy:.4f}")

    # Display the classification report as a table
    st.table(classification_df)

    # Input box for user sentence
    user_sentence = st.text_input("Type your sentence here:")

    if user_sentence:
        # Perform MWE detection
        table_data, detected_mwes = perform_mwe_detection(user_sentence.lower(), model, tokenizer)

        # Display token-level predictions
        st.markdown('<h3 style="color:skyblue;">Token-Level Predictions</h3>', unsafe_allow_html=True)
        st.table(table_data)

        # Highlight detected MWEs
        st.markdown('<h3 style="color:skyblue;">Detected MWEs</h3>', unsafe_allow_html=True)
        if detected_mwes:
            for mwe in detected_mwes:
                st.markdown(f'<span style="color:green; font-weight:bold;">- [{mwe}]</span>', unsafe_allow_html=True)
        else:
            st.info("No MWEs detected in the sentence.")

        # Add some spacing for aesthetics
        st.markdown("---")

elif st.session_state["page"] == "Excel File Prediction":
    st.title("📄 MWEs Detection on Excel File")
    st.write(
        """
        This app detects **Multi-Word Expressions (MWEs)** in an excel file using a fine-tuned 
        **RoBERTa-CRF model**.
        """
    )
    uploaded_file = st.file_uploader("Upload your Excel file:", type=["xlsx"])

    if uploaded_file:
        # Read the uploaded Excel file
        df = pandas.read_excel(uploaded_file)

        # Check if 'Sentence' column exists
        if "Sentence" not in df.columns:
            st.error("The uploaded file must contain a 'Sentence' column.")
        else:
            detected_mwes = []
            predicted_labels = []

            for sentence in df["Sentence"]:
                table_data, mwes = perform_mwe_detection(str(sentence).lower(), model, tokenizer)

                # Extract predictions for the sentence
                labels = [entry["Prediction"] for entry in table_data]
                predicted_labels.append(", ".join(labels))  # Join labels with commas

                # Process MWEs into desired format
                if not mwes:
                    detected_mwes.append("")  # No MWE detected -> Blank
                elif len(mwes) == 1:
                    detected_mwes.append(mwes[0])  # Single MWE -> No trailing comma
                else:
                    detected_mwes.append(", ".join(mwes))  # Multiple MWEs -> Comma-separated

            # Add Predicted Labels and Detected MWEs columns to the DataFrame
            df["Predicted Labels"] = predicted_labels
            df["Detected MWEs"] = detected_mwes

            # Display the updated DataFrame
            st.markdown('<h3 style="color:skyblue;">Updated Excel File</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10))

            # Convert the DataFrame to an Excel file for download
            @st.cache_data
            def convert_df_to_excel(dataframe):
                output = io.BytesIO()
                with pandas.ExcelWriter(output, engine="xlsxwriter") as writer:
                    dataframe.to_excel(writer, index=False, sheet_name="Sheet1")
                processed_file = output.getvalue()
                return processed_file

            # Create and download the Excel file
            excel_file = convert_df_to_excel(df)
            st.download_button(
                label="📥 Download Updated Excel File",
                data=excel_file,
                file_name="Updated_Dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
