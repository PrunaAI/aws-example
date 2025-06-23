import json
import os


import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from pruna import SmashConfig, smash
from transformers import AutoModelForCausalLM, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        """Called once when the model is being loaded."""
        # Load the LLM model and tokenizer
        #model_name = "NousResearch/Llama-3.2-1B"  # You can change this to any LLM
        #model_name = "meta-llama/Llama-3.1-8b-Instruct"  # You can change this to any LLM
        model_path = os.path.join(os.path.dirname(__file__), "model")
        tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)#, cache_dir="/efs/shared/models")
        #self.model = AutoModelForCausalLM.from_pretrained(
        #    model_name, torch_dtype=torch.float16, device_map="auto", #cache_dir="/efs/shared/models",
        #    local_files_only=True
        #)

        # Initialize the SmashConfig
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["hqq_weight_bits"] = 4
        smash_config["hqq_compute_dtype"] = "torch.bfloat16"
        smash_config["compiler"] = "torch_compile"
        smash_config["torch_compile_fullgraph"] = True
        smash_config["torch_compile_dynamic"] = True
        smash_config._prepare_saving = False

        # Smash the model
        self.model = smash(model=self.model, smash_config=smash_config)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to GPU
        self.model = self.model.to("cuda")

        # Parse model configuration
        self.model_config = json.loads(args["model_config"])

        # Get output data type
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_TEXT"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Called for inference requests."""
        responses = []
        for request in requests:
            # Get input text
            input_text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_texts = input_text_tensor.as_numpy().astype(str).flatten().tolist()

            # Get max tokens
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")
            max_tokens = max_tokens_tensor.as_numpy().astype(int).flatten().tolist()

            # Generate text responses
            generated_texts = []
            for i, text in enumerate(input_texts):
                # Tokenize input
                model_inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True
                ).to("cuda")

                # Generate response
                with torch.no_grad():
                    chat_history_ids = self.model.generate(
                        model_inputs["input_ids"],
                        min_new_tokens=max_tokens[i],
                        max_new_tokens=max_tokens[i]
                    )

                # Extract only the response part (after the input)
                response_ids = chat_history_ids[
                    :, model_inputs["input_ids"].shape[-1] :
                ]
                response_text = self.tokenizer.decode(
                    response_ids[0], skip_special_tokens=True
                ).strip()

                # Fallback if response is empty or too short
                if not response_text or len(response_text) < 3:
                    response_text = "I'm not sure how to respond to that."

                generated_texts.append(response_text)

            # Convert the list of texts to a numpy array
            output_array = np.array(generated_texts, dtype=object).reshape(-1, 1)

            # Create Triton output tensor
            output_tensor = pb_utils.Tensor("OUTPUT_TEXT", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """Called when the model is being unloaded."""
        print("Cleaning up LLM model...")
