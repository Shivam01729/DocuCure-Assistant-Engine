from flask import Flask, render_template, request, jsonify
from src.medical_bot.retrieval import get_rag_chain

app = Flask(__name__)

# Initialize the chain once at startup
try:
    rag_chain = get_rag_chain()
    print("RAG Chain initialized successfully.")
except Exception as e:
    print(f"Error initializing RAG Chain: {e}")
    rag_chain = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if not rag_chain:
        return "System initializing or configuration error.", 503
        
    try:
        msg = request.form["msg"]
        # Invoke the chain
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print("Input:", msg)
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print(f"Error processing request: {e}")
        return "An error occurred while processing your request.", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
