"""
demo.py — Flask web demo for the RAG News Assistant.
"""

from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()

from src.ingest import ingest_documents, get_vector_store
from src.generator import rag_pipeline
from src.security import SecurityError, validate_url, sanitise_query

app = Flask(__name__)

def make_page(active, body):
    tabs = {
        "ask": "active" if active == "ask" else "",
        "add": "active" if active == "add" else "",
        "kb":  "active" if active == "kb"  else "",
    }
    return f"""<!DOCTYPE html>
<html>
<head>
<title>RAG News Assistant</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f5f5;padding:24px;margin:0}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{font-size:22px;margin-bottom:4px}}
.sub{{color:#666;font-size:13px;margin-bottom:20px}}
.tabs{{display:flex;gap:6px;margin-bottom:20px}}
.tab{{padding:8px 18px;border:1px solid #ddd;border-radius:6px;text-decoration:none;font-size:13px;color:#111;background:white}}
.tab.active{{background:#111;color:white;border-color:#111}}
.card{{background:white;border:1px solid #ddd;border-radius:8px;padding:24px}}
label{{font-size:13px;font-weight:500;display:block;margin:14px 0 5px}}
input,textarea{{width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;font-size:13px;font-family:inherit;box-sizing:border-box}}
textarea{{resize:vertical}}
button{{margin-top:12px;padding:10px 22px;background:#111;color:white;border:none;border-radius:6px;cursor:pointer;font-size:13px}}
button:hover{{background:#333}}
.box{{margin-top:12px;padding:14px;background:#f9f9f9;border:1px solid #eee;border-radius:6px;font-size:13px;line-height:1.6;white-space:pre-wrap;display:none}}
.stat{{font-size:12px;color:#888;margin-top:8px}}
.loader{{display:none;margin-top:12px;font-size:13px;color:#666}}
.ex{{margin-top:16px}}
.ex span{{font-size:12px;color:#888}}
.eb{{display:inline-block;margin:4px 4px 0 0;padding:5px 10px;background:#f0f0f0;border:1px solid #ddd;border-radius:4px;cursor:pointer;font-size:12px}}
.eb:hover{{background:#e0e0e0}}
hr{{border:none;border-top:1px solid #eee;margin:20px 0}}
</style>
</head>
<body>
<div class="wrap">
<h1>RAG News Assistant</h1>
<p class="sub">Retrieval-Augmented Generation — ask questions, get cited answers from your news knowledge base.</p>
<div class="tabs">
  <a href="/" class="tab {tabs['ask']}">Ask</a>
  <a href="/add" class="tab {tabs['add']}">Add Articles</a>
  <a href="/kb" class="tab {tabs['kb']}">Knowledge Base</a>
</div>
<div class="card">
{body}
</div>
</div>
</body>
</html>"""


ASK_BODY = """
<label>Your Question</label>
<input type="text" id="q" placeholder="What is the EU AI Act?" onkeydown="if(event.key==='Enter')ask()">
<button onclick="ask()">Ask</button>
<div class="loader" id="loader">Thinking...</div>
<div class="ex">
  <span>Try: </span>
  <span class="eb" onclick="setq(this)">What is the maximum fine for violating the AI Act?</span>
  <span class="eb" onclick="setq(this)">What efficiency have solar panels achieved?</span>
  <span class="eb" onclick="setq(this)">What score did GPT-4 get on the bar exam?</span>
</div>
<label>Answer</label>
<div class="box" id="answer"></div>
<label>Sources</label>
<div class="box" id="sources"></div>
<div class="stat" id="stats"></div>
<script>
function setq(el){ document.getElementById('q').value=el.innerText; }
function ask(){
  var q=document.getElementById('q').value.trim();
  if(!q)return;
  document.getElementById('loader').style.display='block';
  document.getElementById('answer').style.display='none';
  document.getElementById('sources').style.display='none';
  fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('loader').style.display='none';
    document.getElementById('answer').style.display='block';
    document.getElementById('sources').style.display='block';
    document.getElementById('answer').innerText=d.answer;
    document.getElementById('sources').innerText=d.sources||'No sources found';
    document.getElementById('stats').innerText=d.stats||'';
  });
}
</script>"""


ADD_BODY = """
<h3 style="font-size:15px;margin-bottom:4px">Add by URL</h3>
<p style="font-size:12px;color:#888;margin-bottom:10px">Allowed: bbc.com, reuters.com, theguardian.com, apnews.com, ft.com</p>
<input type="text" id="url" placeholder="https://www.bbc.com/news/...">
<button onclick="addUrl()">Add URL</button>
<div class="box" id="url-status" style="display:block;display:none"></div>
<hr>
<h3 style="font-size:15px;margin-bottom:10px">Paste Article Text</h3>
<label>Title</label>
<input type="text" id="title" placeholder="e.g. BBC AI Report 2024">
<label>Article Text (min 100 characters)</label>
<textarea id="text" rows="8" placeholder="Paste full article text here..."></textarea>
<button onclick="addText()">Add Text</button>
<div class="box" id="text-status"></div>
<script>
function addUrl(){
  var url=document.getElementById('url').value.trim();
  if(!url)return;
  var s=document.getElementById('url-status');
  s.style.display='block';s.innerText='Fetching...';
  fetch('/api/add-url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:url})})
  .then(r=>r.json()).then(d=>{s.innerText=d.message;});
}
function addText(){
  var title=document.getElementById('title').value.trim();
  var text=document.getElementById('text').value.trim();
  var s=document.getElementById('text-status');
  s.style.display='block';
  if(!title||!text){s.innerText='Please fill in both title and text.';return;}
  s.innerText='Adding...';
  fetch('/api/add-text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title,text:text})})
  .then(r=>r.json()).then(d=>{s.innerText=d.message;});
}
</script>"""


@app.route("/")
def ask_page():
    return make_page("ask", ASK_BODY)

@app.route("/add")
def add_page():
    return make_page("add", ADD_BODY)

@app.route("/kb")
def kb_page():
    try:
        vs = get_vector_store()
        data = vs.get()
        n = len(data.get("ids", []))
        titles = sorted(set(m.get("title","Unknown") for m in data.get("metadatas",[])))
        stats = f"{n} chunks from {len(titles)} articles\n\nArticles:\n" + "\n".join(f"  - {t}" for t in titles)
    except Exception as e:
        stats = f"Error: {e}"
    body = f'<h3 style="font-size:15px;margin-bottom:16px">Current Knowledge Base</h3><div class="box" style="display:block">{stats}</div><button onclick="location.reload()">Refresh</button>'
    return make_page("kb", body)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    question = request.json.get("question","").strip()
    try:
        sanitise_query(question)
        result = rag_pipeline(question)
        return jsonify({
            "answer": result["answer"],
            "sources": "\n".join(result.get("sources",[])),
            "stats": f"{result['num_chunks_retrieved']} chunks | relevance: {result['avg_relevance_score']:.0%} | tokens: {result['tokens_used']}"
        })
    except SecurityError as e:
        return jsonify({"answer": f"Security check failed: {e}", "sources":"", "stats":""})
    except Exception as e:
        return jsonify({"answer": f"Error: {e}", "sources":"", "stats":""})

@app.route("/api/add-url", methods=["POST"])
def api_add_url():
    url = request.json.get("url","").strip()
    try:
        validate_url(url)
        from src.ingest import ingest_urls
        n = ingest_urls([url])
        return jsonify({"message": f"Added {n} chunks from: {url}"})
    except SecurityError as e:
        return jsonify({"message": f"Security check failed: {e}"})
    except Exception as e:
        return jsonify({"message": f"Failed: {e}"})

@app.route("/api/add-text", methods=["POST"])
def api_add_text():
    data = request.json
    title = data.get("title","").strip()
    text = data.get("text","").strip()
    if len(text) < 100:
        return jsonify({"message": "Text too short (minimum 100 characters)"})
    try:
        doc = {"title": title, "text": text, "url": f"manual://{title}", "source": "manual"}
        n = ingest_documents([doc])
        return jsonify({"message": f"Added {n} chunks from: {title}"})
    except Exception as e:
        return jsonify({"message": f"Failed: {e}"})

if __name__ == "__main__":
    print("\nRAG News Assistant running at: http://127.0.0.1:5001")
    print("Open that URL in your browser\n")
    app.run(debug=False, port=5001)
