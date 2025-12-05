#!/usr/bin/env python3
"""
Test /no_think prompt-based approach for QWen models
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def test_no_think_approaches():
    """Test different ways to use /no_think"""
    print("🔄 Testing /no_think Prompt Approach")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ No NVIDIA_API_KEY")
        return
    
    model_name = "qwen/qwen3-235b-a22b"
    model_name = "nvidia/nvidia-nemotron-nano-9b-v2"
    print(f"Testing: {model_name}")
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
            max_tokens=200
        )
        
        base_question = "What is 1+1? Please explain your reasoning."
        base_system = "You are a helpful assistant."
        
        print(f"\n1. Baseline (no /no_think):")
        try:
            messages = [
                SystemMessage(content=base_system),
                HumanMessage(content=base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n2. /no_think in system message:")
        try:
            messages = [
                SystemMessage(content=base_system + " /no_think"),
                HumanMessage(content=base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n3. /no_think at start of user message:")
        try:
            messages = [
                SystemMessage(content=base_system),
                HumanMessage(content="/no_think " + base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n4. /no_think as separate message:")
        try:
            messages = [
                SystemMessage(content=base_system),
                HumanMessage(content="/no_think"),
                HumanMessage(content=base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n5. Empty system + /no_think in user message:")
        try:
            messages = [
                HumanMessage(content="/no_think " + base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\n6. Compare with parameter approach (thinking=False):")
        try:
            messages = [
                SystemMessage(content=base_system),
                HumanMessage(content=base_question)
            ]
            response = llm.invoke(
                messages, 
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
            
        print(f"\n7. Test /think to enable reasoning:")
        try:
            messages = [
                SystemMessage(content=base_system),
                HumanMessage(content="/think " + base_question)
            ]
            response = llm.invoke(messages)
            print(f"   Response: {response.content}")
            print(f"   Length: {len(response.content)}")
        except Exception as e:
            print(f"   Error: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_no_think_approaches()
