from .model import GGUFParse
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.exceptions import SecurityError

class GGUSSecurity(GGUFParse):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.metadata = super().get_metadata()
    
    def is_template_injection(self) -> bool:
        """
        Check for possible Server-Side Template Injection when the .gguf -> chat_template
        is rendered by jinja2.

        Returns:
            bool: If the .gguf file contains Server-Side Template Injection
        """
        chat_template = self.metadata.get('tokenizer.chat_template').get('value')
        print(chat_template)
        if chat_template is None:
            return False
        
        try:
            ImmutableSandboxedEnvironment().from_string(chat_template).render()
            return False
        except SecurityError:
            return True