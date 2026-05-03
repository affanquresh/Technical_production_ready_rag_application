const chat = document.getElementById("chat");

function addMessage(content, sender) {
  const msg = document.createElement("div");
  msg.className = "msg " + sender;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (sender === "bot") {
    bubble.innerHTML = marked.parse(content); // render markdown
  } else {
    bubble.innerText = content;
  }

  msg.appendChild(bubble);
  chat.appendChild(msg);

  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function showTyping() {
  const msg = document.createElement("div");
  msg.className = "msg bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing";
  bubble.innerHTML = `
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  `;

  msg.appendChild(bubble);
  chat.appendChild(msg);

  chat.scrollTop = chat.scrollHeight;
  return msg;
}

async function sendMessage() {
  const input = document.getElementById("query");
  const text = input.value.trim();

  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  const typingMsg = showTyping();

  try {
    const res = await fetch("https://technicalproductionreadyragapplication-production.up.railway.app/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query: text })
    });

    const data = await res.json();

    chat.removeChild(typingMsg);
    addMessage(data.answer, "bot");

  } catch (err) {
    chat.removeChild(typingMsg);
    addMessage("⚠️ Error connecting to backend", "bot");
  }
}

function newChat() {
  chat.innerHTML = "";
}

document.getElementById("query").addEventListener("keypress", function(e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});