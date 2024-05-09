from ggufsecurity import GGUFSecurity

if __name__ == "__main__":
    print(GGUFSecurity('examples/chat_template_malicious_model.gguf').is_template_injection())