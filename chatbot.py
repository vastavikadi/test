from dotenv import load_dotenv
import google.generativeai as genai
import os
from PIL import Image
import pdfplumber

# Load the environment variables from .env file
load_dotenv()

# Configure the Generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Pre-defined prompts for the LifeVault project
pre_made_prompts = {
    "What is LifeVault?": "LifeVault is a secure digital vault for storing and managing personal and sensitive data using blockchain technology.",
    "How does LifeVault ensure my data security?": "LifeVault uses end-to-end encryption and decentralized storage solutions to protect your data from unauthorized access.",
    "What is blockchain-based control?": "Blockchain-based control allows users to manage their data ownership and access rights securely, ensuring transparency and integrity.",
    "How do I access my encrypted data?": "You can access your encrypted data through the LifeVault platform using your secure login credentials and encryption keys.",
    "Can I integrate LifeVault with other apps?": "Yes, LifeVault offers APIs and SDKs for seamless integration with various applications to enhance your data management capabilities.",
    "How does LifeVault use decentralized storage?": "LifeVault utilizes decentralized storage networks to distribute your data securely across multiple nodes, enhancing redundancy and availability.",
    "Is LifeVault suitable for corporate data?": "Absolutely! LifeVault is designed to handle both personal and corporate data, providing tailored solutions for data management.",
    "How do I start using LifeVault?": "To start using LifeVault, sign up on our website, create an account, and follow the onboarding instructions.",
    "Why use blockchain for personal data?": "Using blockchain for personal data ensures enhanced security, transparency, and user control over data ownership and access.",
    "What are the benefits of LifeVault's privacy features?": "LifeVault's privacy features include data encryption, user-controlled access, and the ability to revoke permissions at any time."
}

# Function to get a response from the model based on image input
def image_input(image_path, input_text=""):
    try:
        image = Image.open(image_path)
    except Exception as e:
        return f"Error opening image: {e}"

    # Generate response based on the presence of input_text
    try:
        if input_text.strip() == "":
            response = model.generate_content(image)
        else:
            response = model.generate_content([input_text, image])

        return response.text
    except Exception as e:
        return f"Error generating content for the image: {e}"

# Function to get a response from the model based on document input
def document_input(doc_path):
    pdf_content = "Summarize the content of this document:\n\n"

    try:
        with pdfplumber.open(doc_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:  # Ensure there's text to process
                    pdf_content += f"Page number: {page_num}\n{text}\n"
                else:
                    pdf_content += f"Page number: {page_num}\n[No text extracted from this page]\n"

        # Generate response based on the extracted content
        response = model.generate_content(pdf_content)
        return response.text
    except Exception as e:
        return f"Error processing document: {e}"

# Function to handle pre-made prompts
def handle_premade_prompt(prompt):
    if prompt in pre_made_prompts:
        return pre_made_prompts[prompt]
    else:
        return "Prompt not found."

if __name__ == '__main__':
    # Example usage of image input
    image_response = image_input(image_path="sample_input.jpg")
    print("Image Response:\n", image_response)

    # Example usage of document input
    document_response = document_input("sample_input_doc.pdf")
    print("\nDocument Response:\n", document_response)

    # Example usage of pre-made prompts
    for prompt in pre_made_prompts.keys():
        prompt_response = handle_premade_prompt(prompt)
        print(f"\nResponse for '{prompt}':\n{prompt_response}")
