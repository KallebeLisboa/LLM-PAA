import os
import json 
from huggingface_hub import hf_hub_download  # baixa arquivo por arquivo
                                                
REPO_ID = "MedRAG/pubmed"
OUTPUT_FILE = "pubmed_10gb.jsonl"
TARGET_BYTES = (102 * (1024 ** 3)) // 10  # 10.952.166.604 bytes (~10,2 GB)

    
current_bytes = 0
total_docs = 0 
  
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f: 
    # Baixa dos artigos mais recentes (1166) para trás
    for shard_num in range(1166, 0, -1):
        filename = f"chunk/pubmed23n{shard_num:04d}.jsonl"
        print(f"Baixando {filename}... (Acumulado: {current_bytes / (1024**3):.2f} GB)")
            
        try:
            local_shard = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset"
            ) 
        except Exception as e:
            print(f"Aviso ao baixar {filename}: {e}") 
            continue
  
        with open(local_shard, "r", encoding="utf-8") as in_f: 
            for line in in_f:
                doc = json.loads(line)
                title = doc.get("title", "").strip()
                content = doc.get("content", "").strip() 
                    
                # Filtro de qualidade: exige título e sinopse com pelo menos 250 caracteres
                if not title or len(content) < 250: 
                    continue
                     
                clean_doc = {
                    "id": doc.get("id"), 
                    "PMID": doc.get("PMID"),
                    "title": title,
                    "content": content
                }
                    
                out_line = json.dumps(clean_doc, ensure_ascii=False) + "\n"
                out_f.write(out_line)
                     
                current_bytes += len(out_line.encode("utf-8"))
                total_docs += 1
                    
                if current_bytes >= TARGET_BYTES:
                    break
            
        # Remove tanto o arquivo real (blob) quanto o atalho (symlink) do cache para não gastar SSD em dobro!
        if os.path.exists(local_shard):
            real_blob = os.path.realpath(local_shard)
            os.remove(local_shard)
            if os.path.exists(real_blob):
                os.remove(real_blob)
  
        if current_bytes >= TARGET_BYTES:
            print(f"\nConcluído! {total_docs} artigos salvos ({current_bytes / (1024**3):.2f} GB).")
            break
