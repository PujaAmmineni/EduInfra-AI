// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const pdfBox = document.getElementById('pdf-box');
    const chatBox = document.getElementById('chat-box');

    function appendMessage(text, sender, isTyping = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = sender === 'user' ? '/static/images/user.png' : '/static/images/bot.png';
        avatar.alt = sender === 'user' ? 'User Avatar' : 'Chatbot Avatar';
        messageDiv.appendChild(avatar);

        let messageText = document.createElement('p');
        messageText.classList.add('message-text');

        if (isTyping) {
            messageText.classList.add('typing-indicator');
        } else {
            messageText.innerHTML = marked.parse(text);
            attachLinkHandlers(messageText);
        }

        messageDiv.appendChild(messageText);
        chatContainer.appendChild(messageDiv);
        scrollToBottom();

        return { messageDiv, messageText };
    }

    function attachLinkHandlers(element) {
        const links = element.querySelectorAll('a');
        links.forEach(link => {
            link.addEventListener('click', (event) => {
                event.preventDefault();
                openPDFInBox(link.href);
            });
        });
    }

    function openPDFInBox(url) {
        pdfBox.innerHTML = `
            <button class="close-pdf" onclick="closePDF()">X</button>
            <iframe id="pdfViewer" src="${url}" style="width: 100%; height: 100%;" allowfullscreen></iframe>
        `;
        pdfBox.classList.remove("hidden");
        pdfBox.classList.add("expanded");
        chatBox.classList.add("pdf-open");
    }

    window.closePDF = function() {
        pdfBox.classList.remove("expanded");
        pdfBox.classList.add("hidden");
        chatBox.classList.remove("pdf-open");
    };

    function typeText(element, rawText, speed = 20) {
        let parsedText = marked.parse(rawText);
        let tempDiv = document.createElement("div");
        tempDiv.innerHTML = parsedText; // Convert to HTML

        let textContent = tempDiv.innerText; // Plain text
        let finalHTML = tempDiv.innerHTML;    // Full HTML

        let index = 0;
        function type() {
            if (index < textContent.length) {
                element.innerText = textContent.substring(0, index + 1);
                index++;
                scrollToBottom();
                setTimeout(type, speed);
            } else {
                setTimeout(() => {
                    element.innerHTML = finalHTML;
                    attachLinkHandlers(element);
                    scrollToBottom();
                }, 300);
            }
        }
        type();
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // User message
        appendMessage(message, 'user');
        userInput.value = "";
        scrollToBottom();

        // Show Typing Indicator
        const { messageDiv: typingDiv } = appendMessage("", 'bot', true);

        fetch('/cee_chat', {
            method: 'POST',
            body: JSON.stringify({ user_message: message }),
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => {
            if (!response.ok) {
                // If user is not logged in or an error occurs
                return response.json().then(errData => {
                    throw new Error(errData.bot_reply || "Error");
                });
            }
            return response.json();
        })
        .then(data => {
            setTimeout(() => {
                chatContainer.removeChild(typingDiv);
                const { messageText } = appendMessage("", 'bot');
                typeText(messageText, data.bot_reply);
            }, 1000);
        })
        .catch(error => {
            chatContainer.removeChild(typingDiv);
            appendMessage("Error: " + error.message, 'bot');
            console.error('Error:', error);
        });
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: "smooth"
        });
    }
});

// Attach the beforeunload listener (outside or inside the DOMContentLoaded callback)
window.addEventListener('beforeunload', function() {
    navigator.sendBeacon('/logout');
});