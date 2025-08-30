#!/usr/bin/env python3
"""
Test notebook for disabling thinking mode across different models
Testing: QWen 3, Gemini 2.5 Flash, OpenAI, and DeepSeek
Question: "What is 1+1?" (simple to see thinking vs direct response)
"""

import os
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

# Configuration
SYSTEM_MESSAGE = "You are a helpful assistant."
TEST_QUESTION = "What is 1+1? Please explain your reasoning."
STOP_SEQUENCE = None

def test_model(model_name, llm_instance, extra_params=None):
    """Test a model with and without thinking mode"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    messages = [
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=TEST_QUESTION)
    ]
    
    # Test 1: Default behavior (thinking enabled)
    print(f"\n--- Test 1: Default behavior (thinking enabled) ---")
    try:
        response = llm_instance.invoke(messages, stop=STOP_SEQUENCE)
        print(f"Response: {response.content}")
        print(f"Response length: {len(response.content)} chars")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Thinking disabled
    print(f"\n--- Test 2: Thinking disabled ---")
    try:
        invoke_kwargs = {"stop": STOP_SEQUENCE}
        if extra_params:
            invoke_kwargs.update(extra_params)
        
        response = llm_instance.invoke(messages, **invoke_kwargs)
        print(f"Response: {response.content}")
        print(f"Response length: {len(response.content)} chars")
    except Exception as e:
        print(f"Error: {e}")

# Test Configuration for different models
def test_qwen3():
    """Test QWen 3 models via HuggingFace"""
    print("🧠 Testing QWen 3 Models")
    
    # List of QWen models to try (including older versions that might be available)
    qwen_models = [
        "Qwen/Qwen3-8B",
        #"Qwen/Qwen3-4B",
        #"Qwen/Qwen3-30B-A3B",
        #"Qwen/Qwen3-32B",
        #"Qwen/Qwen3-14B",
        "Qwen/Qwen3-235B-A22B"
    ]
    
    for model_name in qwen_models:
        print(f"\n--- Testing {model_name} ---")
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
                openai_api_base="https://router.huggingface.co/v1",
                max_tokens=512
            )
            
            messages = [
                SystemMessage(content=SYSTEM_MESSAGE),
                HumanMessage(content=TEST_QUESTION)
            ]
            
            # Test 1: Default behavior
            print("  Default response:")
            try:
                response = llm.invoke(messages)
                print(f"  ✅ Response: {response.content[:200]}...")
                print(f"  Response length: {len(response.content)} chars")
                
                # Test 2: With thinking disabled (now that we know the model works)
                print("  With thinking disabled:")
                try:
                    response_no_think = llm.invoke(
                        messages,
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        }
                    )
                    print(f"  ✅ No-think response: {response_no_think.content[:200]}...")
                    print(f"  No-think length: {len(response_no_think.content)} chars")
                    
                    # Compare lengths
                    if len(response.content) != len(response_no_think.content):
                        print(f"  🎯 SUCCESS: Thinking toggle works! Length diff: {len(response.content) - len(response_no_think.content)} chars")
                    else:
                        print(f"  ⚠️ No difference in response length - thinking toggle may not work")
                        
                except Exception as e:
                    print(f"  ⚠️ Thinking disable not supported: {e}")
                
                # Success with this model, stop testing others
                print(f"  ✅ Successfully tested {model_name}")
                return True
                
            except Exception as e:
                print(f"  ❌ Default error: {e}")
                continue
            
        except Exception as e:
            print(f"  ❌ Model {model_name} failed: {e}")
            continue
    
    print("  ❌ No QWen models were accessible via HuggingFace")
    return False

def test_gemini_25():
    """Test Gemini 2.5 Flash with different approaches"""
    print("🔷 Testing Gemini 2.5 Flash")
    
    try:
        # Create Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=512,
            convert_system_message_to_human=True
        )
        
        # Gemini doesn't support extra_body in LangChain
        # Instead, we can try different model configurations
        print(f"\n--- Gemini 2.5 Flash Default ---")
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=TEST_QUESTION)
        ]
        
        try:
            response = llm.invoke(messages)
            print(f"Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
        except Exception as e:
            print(f"Error: {e}")
            
        # Try with shorter prompt to discourage reasoning
        print(f"\n--- Gemini 2.5 Flash with Direct Prompt ---")
        short_messages = [
            HumanMessage(content="What is 1+1? Answer briefly.")
        ]
        
        try:
            response = llm.invoke(short_messages)
            print(f"Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
        except Exception as e:
            print(f"Error: {e}")
        
    except Exception as e:
        print(f"Failed to test Gemini 2.5: {e}")

def test_openai_o1():
    """Test OpenAI o1 models (they don't support system messages)"""
    print("🤖 Testing OpenAI o1-mini")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("  ❌ No OpenAI API key available")
        return
    
    try:
        # Create OpenAI o1 model with correct parameters
        print(f"\n--- OpenAI o1-mini (with reasoning) ---")
        try:
            from langchain_openai import ChatOpenAI as ChatOpenAI_New
            
            llm = ChatOpenAI_New(
                model="o1-mini",
                temperature=1,  # o1 models have fixed temperature
                max_completion_tokens=512  # Use max_completion_tokens instead of max_tokens
            )
            
            # o1 models don't support system messages, only user messages
            messages = [
                HumanMessage(content=TEST_QUESTION)  # No system message
            ]
            
            response = llm.invoke(messages)
            print(f"Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
            
        except ImportError:
            print("  ⚠️ langchain_openai not available, trying legacy approach")
            # Fallback to old ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use a model that works
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=512
            )
            
            messages = [
                HumanMessage(content=TEST_QUESTION)
            ]
            
            response = llm.invoke(messages)
            print(f"GPT-4o-mini Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
        
    except Exception as e:
        print(f"Failed to test OpenAI models: {e}")

def test_simple_models():
    """Test simple models that should definitely work"""
    print("🔥 Testing Simple Working Models")
    
    # Test regular OpenAI GPT models
    if os.getenv("OPENAI_API_KEY"):
        print(f"\n--- OpenAI GPT-4o-mini (baseline) ---")
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=256
            )
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 1+1? Be brief.")
            ]
            
            response = llm.invoke(messages)
            print(f"✅ GPT-4o-mini Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
            
        except Exception as e:
            print(f"❌ GPT-4o-mini Error: {e}")
    
    # Test Gemini (we know this works)
    if os.getenv("GOOGLE_API_KEY"):
        print(f"\n--- Gemini 2.0 Flash (baseline) ---")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
                max_output_tokens=256,
                convert_system_message_to_human=True
            )
            
            messages = [
                HumanMessage(content="What is 1+1? Be very brief.")
            ]
            
            response = llm.invoke(messages)
            print(f"✅ Gemini Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
            
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
    """Test DeepSeek models (via model name switching)"""
    print("🧮 Testing DeepSeek R1")
    
    try:
        # Test 1: DeepSeek Reasoner (with thinking)
        print(f"\n--- DeepSeek R1 with reasoning ---")
        llm_reasoner = ChatOpenAI(
            model="deepseek-reasoner",
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY") or "dummy-key",
            openai_api_base="https://api.deepseek.com/v1",
            max_tokens=512
        )
        
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=TEST_QUESTION)
        ]
        
        try:
            response = llm_reasoner.invoke(messages)
            print(f"Reasoner Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
        except Exception as e:
            print(f"Reasoner Error: {e}")
        
        # Test 2: DeepSeek Chat (without thinking)  
        print(f"\n--- DeepSeek R1 without reasoning (chat model) ---")
        llm_chat = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY") or "dummy-key",
            openai_api_base="https://api.deepseek.com/v1", 
            max_tokens=512
        )
        
        try:
            response = llm_chat.invoke(messages)
            print(f"Chat Response: {response.content}")
            print(f"Response length: {len(response.content)} chars")
        except Exception as e:
            print(f"Chat Error: {e}")
            
    except Exception as e:
        print(f"Failed to test DeepSeek: {e}")

def test_nvidia_nim_qwen():
    """Test QWen via NVIDIA NIM (alternative approach)"""
    print("🚀 Testing QWen 3 via NVIDIA NIM")
    
    try:
        # Create QWen model via NVIDIA NIM
        llm = ChatOpenAI(
            model="qwen/qwen3-235b-a22b",
            temperature=0,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
            max_tokens=512
        )
        
        # Test with thinking mode disabled
        extra_params = {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        }
        
        test_model("QWen 3 via NVIDIA NIM", llm, extra_params)
        
    except Exception as e:
        print(f"Failed to test QWen via NVIDIA NIM: {e}")

#!/usr/bin/env python3
"""
Diagnostic test to see what QWen 3 is actually returning
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def test_qwen_diagnostic():
    """Detailed diagnostic of QWen 3 responses"""
    print("🔍 QWen 3 Diagnostic Test")

    model_name = "Qwen/Qwen3-235B-A22B" # "Qwen/Qwen3-8B"
    print(f"Testing model: {model_name}")
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
            openai_api_base="https://router.huggingface.co/v1",
            max_tokens=100
        )
        
        # Simple test first
        print(f"\n--- Test 1: Simple Question ---")
        simple_messages = [
            HumanMessage(content="What is 1+1?")
        ]
        
        try:
            response = llm.invoke(simple_messages)
            print(f"Raw response object: {response}")
            print(f"Response content: '{response.content}'")
            print(f"Response type: {type(response.content)}")
            print(f"Response length: {len(response.content)}")
            print(f"Response repr: {repr(response.content)}")
        except Exception as e:
            print(f"Simple test error: {e}")
            return
        
        # Test with thinking disabled
        print(f"\n--- Test 2: With thinking disabled ---")
        try:
            response_no_think = llm.invoke(
                simple_messages,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            print(f"No-think response content: '{response_no_think.content}'")
            print(f"No-think response length: {len(response_no_think.content)}")
            print(f"No-think repr: {repr(response_no_think.content)}")
        except Exception as e:
            print(f"No-think test error: {e}")
        
        # Test with system message
        print(f"\n--- Test 3: With System Message ---")
        system_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 1+1? Be brief.")
        ]
        
        try:
            response_sys = llm.invoke(system_messages)
            print(f"System msg response: '{response_sys.content}'")
            print(f"System msg length: {len(response_sys.content)}")
        except Exception as e:
            print(f"System message test error: {e}")
        
        # Test different model variations
        print(f"\n--- Test 4: Different Model Names ---")
        other_models = [
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-14B"
        ]

        for alt_model in other_models:
            try:
                print(f"  Testing {alt_model}:")
                alt_llm = ChatOpenAI(
                    model=alt_model,
                    temperature=0,
                    openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
                    openai_api_base="https://router.huggingface.co/v1",
                    max_tokens=50
                )
                
                alt_response = alt_llm.invoke([HumanMessage(content="What is 2+2?")])
                print(f"    Response: '{alt_response.content}' (len: {len(alt_response.content)})")
                
                if len(alt_response.content) > 0:
                    print(f"    ✅ {alt_model} works!")
                    break
                    
            except Exception as e:
                print(f"    ❌ {alt_model} failed: {e}")
        
    except Exception as e:
        print(f"Overall test failed: {e}")


def main():
    """Run all model tests focused on QWen 3 and DeepSeek"""
    print("🧪 Testing Thinking Mode Disable: QWen 3 & DeepSeek Focus")
    print("Question:", TEST_QUESTION)
    print("Goal: Compare response length and content with/without thinking")
    
    # Check API keys
    print(f"\n🔑 API Keys Status:")
    print(f"HUGGINGFACE_API_KEY: {'✅' if os.getenv('HUGGINGFACE_API_KEY') else '❌'}")
    print(f"GOOGLE_API_KEY: {'✅' if os.getenv('GOOGLE_API_KEY') else '❌'}")  
    print(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    
    # Run focused tests on QWen 3 and DeepSeek only
    test_simple_models()  # Start with models we know work
    test_qwen3()          # QWen 3 models 
    # test_deepseek_hf()    # DeepSeek models via HuggingFace
    # test_openai_o1()      # OpenAI o1 for comparison
    
    print(f"\n{'='*60}")
    print("🏁 Focused tests completed!")
    print("Key findings to look for:")
    print("• QWen models: Does enable_thinking=False work?")
    print("• DeepSeek R1: Does enable_reasoning=False work?") 
    print("• Response length differences between thinking/no-thinking")
    print(f"{'='*60}")

def test_nvidia_nim_direct():
    """Test NVIDIA NIM using direct OpenAI client (like their example)"""
    print("🚀 Testing NVIDIA NIM with Direct OpenAI Client")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("  ❌ No NVIDIA API key available")
        return
    
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        
        # Test available models first
        print("\n--- Available Models ---")
        try:
            models = client.models.list()
            qwen_models = [m.id for m in models.data if 'qwen' in m.id.lower()]
            print(f"QWen models available: {qwen_models}")
            
            if not qwen_models:
                print("  ❌ No QWen models found")
                return
                
            test_model = qwen_models[0]  # Use first available QWen model
            
        except Exception as e:
            print(f"  ⚠️ Could not list models: {e}")
            test_model = "qwen/qwen3-235b-a22b"  # Default from example
        
        print(f"\n--- Testing Model: {test_model} ---")
        
        # Test 1: With thinking enabled (default)
        print("\n  🧠 Test 1: With thinking enabled")
        try:
            completion = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "What is 1+1? Please explain your reasoning."}],
                temperature=0.2,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=False  # Non-streaming for easier testing
            )
            
            response = completion.choices[0].message.content
            print(f"    Response: {response[:200]}...")
            print(f"    Length: {len(response)} chars")
            print(f"    Has <think>: {'<think>' in response}")
            
        except Exception as e:
            print(f"    ❌ Thinking=True failed: {e}")
            return
        
        # Test 2: With thinking disabled
        print("\n  🚫 Test 2: With thinking disabled")
        try:
            completion_no_think = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "What is 1+1? Please explain your reasoning."}],
                temperature=0.2,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False
            )
            
            response_no_think = completion_no_think.choices[0].message.content
            print(f"    Response: {response_no_think[:200]}...")
            print(f"    Length: {len(response_no_think)} chars")
            print(f"    Has <think>: {'<think>' in response_no_think}")
            
            # Compare results
            if len(response) > len(response_no_think):
                diff = len(response) - len(response_no_think)
                print(f"    🎯 SUCCESS: Thinking toggle works! Difference: {diff} chars")
            else:
                print(f"    ⚠️ No significant difference in response length")
                
        except Exception as e:
            print(f"    ❌ Thinking=False failed: {e}")
        
        # Test 3: Try enable_thinking parameter (alternative)
        print("\n  🔄 Test 3: Try enable_thinking parameter")
        try:
            completion_alt = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "What is 1+1?"}],
                temperature=0.2,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=False
            )
            
            response_alt = completion_alt.choices[0].message.content
            print(f"    Response: {response_alt}")
            print(f"    Length: {len(response_alt)} chars")
            
        except Exception as e:
            print(f"    ❌ enable_thinking=False failed: {e}")
            
    except Exception as e:
        print(f"  ❌ NVIDIA NIM direct test failed: {e}")

def test_nvidia_nim_langchain():
    """Test NVIDIA NIM using LangChain ChatOpenAI"""
    print("\n🔗 Testing NVIDIA NIM with LangChain ChatOpenAI")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("  ❌ No NVIDIA API key available")
        return
    
    try:
        # Test model
        model_name = "qwen/qwen3-235b-a22b"
        print(f"Testing model: {model_name}")
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
            max_tokens=512
        )
        
        messages = [
            HumanMessage(content="What is 1+1? Please explain your reasoning.")
        ]
        
        # Test 1: Default behavior
        print("\n  Default response:")
        try:
            response = llm.invoke(messages)
            print(f"    Response: {response.content[:200]}...")
            print(f"    Length: {len(response.content)} chars")
            print(f"    Has <think>: {'<think>' in response.content}")
        except Exception as e:
            print(f"    ❌ Default test failed: {e}")
            return
        
        # Test 2: With thinking disabled using extra_body
        print("\n  With thinking=False:")
        try:
            response_no_think = llm.invoke(
                messages,
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
            print(f"    Response: {response_no_think.content[:200]}...")
            print(f"    Length: {len(response_no_think.content)} chars")
            print(f"    Has <think>: {'<think>' in response_no_think.content}")
            
            # Compare
            if len(response.content) > len(response_no_think.content):
                diff = len(response.content) - len(response_no_think.content)
                print(f"    🎯 SUCCESS: LangChain thinking toggle works! Difference: {diff} chars")
            else:
                print(f"    ⚠️ No significant difference via LangChain")
                
        except Exception as e:
            print(f"    ❌ LangChain thinking=False failed: {e}")
        
        # Test 3: Alternative parameter name
        print("\n  With enable_thinking=False:")
        try:
            response_alt = llm.invoke(
                messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            print(f"    Alt response: {response_alt.content}")
            print(f"    Alt length: {len(response_alt.content)} chars")
        except Exception as e:
            print(f"    ❌ enable_thinking failed: {e}")
            
    except Exception as e:
        print(f"  ❌ NVIDIA NIM LangChain test failed: {e}")

def main_nvidia():
    """Run all NVIDIA NIM tests"""
    print("🧪 NVIDIA NIM QWen 3 Thinking Mode Control Test")
    print("Goal: Test if NVIDIA NIM supports thinking mode toggle")
    
    # Check API key
    print(f"\n🔑 NVIDIA_API_KEY: {'✅' if os.getenv('NVIDIA_API_KEY') else '❌'}")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ Please set NVIDIA_API_KEY in your .env file")
        return
    
    # Run tests
    test_nvidia_nim_direct()      # Direct OpenAI client (their example style)
    test_nvidia_nim_langchain()   # LangChain approach (your current style)
    
    print(f"\n{'='*60}")
    print("🏁 NVIDIA NIM tests completed!")
    print("Key findings:")
    print("• Does NVIDIA NIM support thinking=True/False?")
    print("• Does it work with LangChain's extra_body parameter?")
    print("• What's the response difference with/without thinking?")
    print(f"{'='*60}")
    
if __name__ == "__main__":
    # main()
    # test_qwen_diagnostic()
    main_nvidia()