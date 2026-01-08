#!/usr/bin/env python3
"""
Simple diagnostic to understand NVIDIA NIM behavior
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

def simple_nvidia_test():
    """Simple test to understand the behavior"""
    print("🔍 Simple NVIDIA NIM Diagnostic")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ No NVIDIA_API_KEY")
        return
    
    model_name = "qwen/qwen3-235b-a22b"
    print(f"Testing: {model_name}")
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
            max_tokens=100  # Small limit for testing
        )
        
        simple_question = "What is 2+2?"
        messages = [HumanMessage(content=simple_question)]
        
        print(f"\n1. Basic test:")
        try:
            response = llm.invoke(messages)
            print(f"   Raw response: {repr(response.content)}")
            print(f"   Length: {len(response.content)}")
            print(f"   Type: {type(response.content)}")
        except Exception as e:
            print(f"   ❌ Basic test failed: {e}")
            return
        
        print(f"\n2. With thinking=True (explicit):")
        try:
            response_think = llm.invoke(
                messages,
                extra_body={"chat_template_kwargs": {"thinking": True}}
            )
            print(f"   Response: {repr(response_think.content[:100])}")
            print(f"   Length: {len(response_think.content)}")
            print(f"   Has <think>: {'<think>' in response_think.content}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n3. With thinking=False:")
        try:
            response_no_think = llm.invoke(
                messages,
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
            print(f"   Response: {repr(response_no_think.content[:100])}")
            print(f"   Length: {len(response_no_think.content)}")
            print(f"   Has <think>: {'<think>' in response_no_think.content}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n4. Test different parameter names:")
        for param_name in ["enable_thinking", "reasoning", "think"]:
            try:
                print(f"   Trying {param_name}=False:")
                response_alt = llm.invoke(
                    messages,
                    extra_body={"chat_template_kwargs": {param_name: False}}
                )
                print(f"     Response: {response_alt.content}")
                print(f"     Length: {len(response_alt.content)}")
            except Exception as e:
                print(f"     Error with {param_name}: {e}")
        
        print(f"\n5. Test without extra_body (pure default):")
        try:
            response_pure = llm.invoke(messages)
            print(f"   Pure default: {repr(response_pure.content)}")
            print(f"   Length: {len(response_pure.content)}")
        except Exception as e:
            print(f"   Pure default error: {e}")
            
    except Exception as e:
        print(f"❌ Overall test failed: {e}")

if __name__ == "__main__":
    simple_nvidia_test()
