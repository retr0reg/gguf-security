#!/usr/bin/env python3
from typing import Any
from gguf import GGUFReader, GGUFValueType  
import numpy as np

class GGUFParse:
    def __init__(
            self,
            model_name
            ):
        
        self.model_name = model_name
        self.reader = GGUFReader(model_name, 'r')

    def get_data(
            self,
            no_tensors: bool = True,
            ) -> list:
        
        def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
            host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
            if reader.byte_order == 'S':
                file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
            else:
                file_endian = host_endian
            return (host_endian, file_endian)
        
        _, file_endian = get_file_host_endian(self.reader)
        metadata: dict[str, Any] = {}
        tensors: dict[str, Any] = {}
        result = {
            "filename": self.model_name,
            "endian": file_endian,
            "metadata": metadata,
            "tensors": tensors,
        }
        for idx, field in enumerate(self.reader.fields.values()):
            curr: dict[str, Any] = {
                "index": idx,
                "type": field.types[0].name if field.types else 'UNKNOWN',
                "offset": field.offset,
            }
            metadata[field.name] = curr
            if field.types[:1] == [GGUFValueType.ARRAY]:
                curr["array_types"] = [t.name for t in field.types][1:]
                itype = field.types[-1]
                if itype == GGUFValueType.STRING:
                    curr["value"] = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
                else:
                    curr["value"] = [pv for idx in field.data for pv in field.parts[idx].tolist()]
            elif field.types[0] == GGUFValueType.STRING:
                curr["value"] = str(bytes(field.parts[-1]), encoding="utf-8")
            else:
                curr["value"] = field.parts[-1].tolist()[0]
        if not no_tensors:
            for idx, tensor in enumerate(self.reader.tensors):
                tensors[tensor.name] = {
                    "index": idx,
                    "shape": tensor.shape.tolist(),
                    "type": tensor.tensor_type.name,
                    "offset": tensor.field.offset,
                }
                
        return result
    
    def get_metadata(self):
        return self.get_data(self.model_name)['metadata']