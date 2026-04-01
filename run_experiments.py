from llama_cpp import Llama
import pandas as pd
from tqdm import tqdm
import json
import time
from datetime import datetime
import gc

# Configuration
MODELS = {
    'fp16': {
        'path': './models/mistral-7b.gguf',
        'n_gpu_layers': 16  # Adjust based on Day 1 findings
    },
    # 'q8_0': {
    #     'path': './models/mistral-7b.Q8_0.gguf',
    #     'n_gpu_layers': 31
    # },
    # 'q4_k_m': {
    #     'path': './models/mistral-7b.Q4_K_M.gguf',
    #     'n_gpu_layers': -1
    # }
}

GENERATION_CONFIG = {
    'max_tokens': 50,
    'temperature': 0.0,
    'top_p': 0.9,
    'top_k': 40,
    'repeat_penalty': 1.1,
    # 'stop': ["</s>", "[INST]", "\n\nHuman:", "\n\nUser:"] # manula chat template
}

# def format_prompt(instruction):
#     """Format prompt for Mistral-Instruct"""
#     return f"[INST] {instruction} [/INST]"

def load_model(config):
    """Load model with specified config"""
    return Llama(
        model_path=config['path'],
        n_gpu_layers=config['n_gpu_layers'],
        n_ctx=2048,
        verbose=False
    )

""" Using Manual chat template"""
# def generate_response(model, prompt):
#     """Generate response with error handling"""
#     try:
#         output = model(
#             prompt,
#             **GENERATION_CONFIG
#         )
#         return {
#             'text': output['choices'][0]['text'].strip(),
#             'tokens': output['usage']['completion_tokens'],
#             'error': None
#         }
#     except Exception as e:
#         return {
#             'text': None,
#             'tokens': 0,
#             'error': str(e)
#         }

def generate_response(model, prompt):
    """Generate response using the robust Chat Completion API"""
    try:
        output = model.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            **GENERATION_CONFIG
        )
        return {
            # Note the different dictionary path for chat completions
            'text': output['choices'][0]['message']['content'].strip(),
            'tokens': output['usage']['completion_tokens'],
            'error': None
        }
    except Exception as e:
        return {
            'text': None,
            'tokens': 0,
            'error': str(e)
        }


def run_experiment(model_name, prompts_df, save_freq=100):
    """Run experiment for one model variant"""
    print(f"\n{'='*60}")
    print(f"Running experiments for: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model...")
    start_load = time.time()
    model = load_model(MODELS[model_name])
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")
    
    results = []
    start_time = time.time()
    
    for idx, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc=f"{model_name}"):
        prompt_text = row['goal']
        # formatted_prompt = format_prompt(prompt_text) 
        """for manual"""
        
        # Generate
        gen_start = time.time()

        #response = generate_response(model, formatted_prompt)
        response = generate_response(model, prompt_text)

        gen_time = time.time() - gen_start
        
        result = {
            'model': model_name,
            'prompt_id': idx,
            'prompt': prompt_text,
            'response': response['text'],
            'tokens_generated': response['tokens'],
            'generation_time': gen_time,
            'error': response['error'],
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Save incrementally
        if len(results) % save_freq == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(f'./results/{model_name}_partial.csv', index=False)
            print(f"  Saved {len(results)} results...")
    
    # Final save
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'./results/{model_name}_complete.csv', index=False)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(prompts_df)
    
    print(f"\nCompleted {model_name}:")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Avg per prompt: {avg_time:.2f}s")
    print(f"  Results saved to: ./results/{model_name}_complete.csv")
    
    # Clean up
    del model
    gc.collect()
    
    return df_results

def main():
    # Load prompts
    prompts_df = pd.read_csv('./data/advbench_full.csv')
    print(f"Loaded {len(prompts_df)} prompts")
    
    # Run experiments for each model
    all_results = {}
    for model_name in [ 'fp16']:  # Start with fp16, add q8_0 and q4_k_m after sanity check)
        results = run_experiment(model_name, prompts_df)
        all_results[model_name] = results
        
        
        print("\nTaking 30s break before next model...")
        time.sleep(30)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()