
const form = document.getElementById("chat-form");
const input = document.getElementById("question-input");
const messages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");

function adicionarMensagem(texto, tipo) {
    const welcome = document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }

    const message = document.createElement("div");
    message.classList.add("message", tipo);

    const content = document.createElement("div");
    content.classList.add("message-content");
    content.textContent = texto;

    message.appendChild(content);
    messages.appendChild(message);

    messages.scrollTop = messages.scrollHeight;

    return message;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const pergunta = input.value.trim();

    if (!pergunta) return;

    adicionarMensagem(pergunta, "user");

    input.value = "";
    sendButton.disabled = true;

    const loading = adicionarMensagem(
        "Pesquisando artigos e preparando a resposta...",
        "assistant"
    );

    try {
        const response = await fetch("/perguntar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                pergunta: pergunta,
                metodo_busca: "hnsw",
                modelo: "phi3:mini",
                top_k: 1
            })
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP: ${response.status}`);
        }

        const data = await response.json();

        loading.querySelector(".message-content").textContent =
            data.resposta;

    } catch (error) {
        loading.querySelector(".message-content").textContent =
            "Não foi possível obter uma resposta. Verifique se a API está funcionando.";

        console.error(error);

    } finally {
        sendButton.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
});

// Enviar com Enter e quebrar linha com Shift + Enter.
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});
