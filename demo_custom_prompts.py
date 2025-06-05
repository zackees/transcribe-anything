#!/usr/bin/env python3
"""
Demo script showing how to use custom prompts with transcribe-anything
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_cli_usage():
    """Demonstrate CLI usage examples"""
    print("=== CLI Usage Examples ===\n")
    
    print("1. Basic transcription with custom prompt:")
    print("   transcribe-anything video.mp4 --initial_prompt \"The speaker discusses AI, machine learning, and neural networks.\"")
    print()
    
    print("2. Using a prompt file:")
    print("   transcribe-anything video.mp4 --prompt_file examples/prompts/technical_terms.txt")
    print()
    
    print("3. Combining with other options:")
    print("   transcribe-anything video.mp4 --device insane --model large-v3 --initial_prompt \"Technical discussion about PyTorch, TensorFlow, and deep learning.\"")
    print()

def demo_python_api():
    """Demonstrate Python API usage"""
    print("=== Python API Examples ===\n")
    
    print("1. Direct prompt usage:")
    print("""
from transcribe_anything import transcribe

result = transcribe(
    url_or_file="video.mp4",
    initial_prompt="The speaker discusses artificial intelligence, machine learning, and neural networks."
)
""")
    
    print("2. Loading prompt from file:")
    print("""
with open("examples/prompts/technical_terms.txt", "r") as f:
    prompt = f.read()

result = transcribe(
    url_or_file="video.mp4",
    initial_prompt=prompt,
    model="large-v3",
    device="insane"
)
""")
    
    print("3. Domain-specific prompts:")
    print("""
# For medical content
medical_prompt = "The discussion covers medical terminology including diagnosis, treatment, symptoms, pathology, anatomy, physiology."

result = transcribe(
    url_or_file="medical_lecture.mp4",
    initial_prompt=medical_prompt
)

# For business content  
business_prompt = "The presentation mentions companies like Microsoft, Google, Apple, Amazon, and discusses AI, cloud computing, and enterprise software."

result = transcribe(
    url_or_file="business_meeting.mp4", 
    initial_prompt=business_prompt
)
""")

def demo_prompt_examples():
    """Show example prompts for different domains"""
    print("=== Example Prompts by Domain ===\n")
    
    print("🤖 Technology/AI:")
    print("   \"The speaker discusses artificial intelligence, machine learning, neural networks, deep learning, PyTorch, TensorFlow, OpenAI, transformers, and large language models.\"")
    print()
    
    print("🏥 Medical/Healthcare:")
    print("   \"The discussion covers medical terminology including diagnosis, treatment, symptoms, pathology, anatomy, physiology, cardiology, neurology, and patient care.\"")
    print()
    
    print("💼 Business/Finance:")
    print("   \"The presentation mentions companies like Microsoft, Google, Apple, Amazon, discusses quarterly earnings, revenue, market share, and business strategy.\"")
    print()
    
    print("🎓 Academic/Research:")
    print("   \"The lecture covers research methodology, statistical analysis, peer review, academic publishing, and scientific methodology.\"")
    print()
    
    print("🎵 Music/Entertainment:")
    print("   \"The discussion includes musical terms, artist names, album titles, genres like jazz, classical, rock, and music production techniques.\"")
    print()

def show_prompt_files():
    """Show the contents of example prompt files"""
    print("=== Example Prompt Files ===\n")
    
    prompt_files = [
        "examples/prompts/technical_terms.txt",
        "examples/prompts/medical_terms.txt", 
        "examples/prompts/business_names.txt"
    ]
    
    for file_path in prompt_files:
        if os.path.exists(file_path):
            print(f"📄 {file_path}:")
            with open(file_path, 'r') as f:
                content = f.read()
                # Show first 200 characters
                if len(content) > 200:
                    print(f"   {content[:200]}...")
                else:
                    print(f"   {content}")
            print()
        else:
            print(f"📄 {file_path}: (file not found)")
            print()

def main():
    print("🎙️  Custom Prompts Demo for transcribe-anything\n")
    print("This demo shows how to use custom prompts to improve transcription accuracy")
    print("for domain-specific vocabulary, names, and technical terms.\n")
    
    demo_cli_usage()
    demo_python_api()
    demo_prompt_examples()
    show_prompt_files()
    
    print("=== Benefits of Custom Prompts ===\n")
    print("✅ Better recognition of technical terms")
    print("✅ Improved accuracy for proper names")
    print("✅ Domain-specific vocabulary support")
    print("✅ Reduced transcription errors")
    print("✅ Works with all Whisper backends (cpu, cuda, insane, mps)")
    print()
    
    print("=== Next Steps ===\n")
    print("1. Try transcribing a file with a custom prompt")
    print("2. Create your own prompt file for your domain")
    print("3. Test with and without prompts to measure improvement")
    print("4. Share your results and contribute example prompts!")

if __name__ == "__main__":
    main()
