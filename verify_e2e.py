import os, torch
from transformers import AutoModel

model_dir = "/home-local/lix88/Projects/pathology_llm/app_v2/model/paige-ai_Prism"
print("Files:", [f for f in os.listdir(model_dir) if f.endswith(".py")])
m = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
print("Loaded:", type(m))

# load sample embeddings from the repo's tcga folder if you downloaded them
pth = os.path.join(model_dir, "tcga", "TCGA-B6-A0WZ-...C6.pth")  # fill in actual name
if os.path.exists(pth) and torch.cuda.is_available():
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        em = torch.load(pth)["embeddings"].unsqueeze(0).cuda()
        reprs = m.slide_representations(em)
        ids = m.generate(key_value_states=reprs["image_latents"], do_sample=False, num_beams=2)
        print(m.untokenize(ids)[0])
