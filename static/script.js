// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const pdfBox = document.getElementById('pdf-box');
    const chatBox = document.getElementById('chat-box');

    // NEW: track current typing controller (null when idle)
    let activeTypeController = null;

    // choose endpoint based on page (/cee uses /cee_chat)
    const CHAT_ENDPOINT = location.pathname.startsWith('/cee') ? '/cee_chat' : '/chat';

    function updateSendButton() {
        if (activeTypeController) {
            sendButton.textContent = 'Stop';
            sendButton.title = 'Stop typing';
            sendButton.classList.add('stop-mode');
        } else {
            sendButton.textContent = '➤';
            sendButton.title = 'Send';
            sendButton.classList.remove('stop-mode');
        }
    }

    function appendMessage(text, sender, isTyping = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = sender === 'user' ? '/static/images/user.png' : '/static/images/bot.png';
        avatar.alt = sender === 'user' ? 'User Avatar' : 'Chatbot Avatar';
        messageDiv.appendChild(avatar);

        const contentWrap = document.createElement('div');
        contentWrap.classList.add('message-content-wrap');
        messageDiv.appendChild(contentWrap);

        let messageText = document.createElement('p');
        messageText.classList.add('message-text');

        if (isTyping) {
            messageText.classList.add('typing-indicator');
        } else {
            messageText.innerHTML = marked.parse(text);
            attachLinkHandlers(messageText);
        }

        contentWrap.appendChild(messageText);

        let suggestionMount = null;
        if (sender === 'bot') {
            suggestionMount = document.createElement('div');
            suggestionMount.classList.add('suggestions');
            contentWrap.appendChild(suggestionMount);
        }

        chatContainer.appendChild(messageDiv);
        scrollToBottom();
        return { messageDiv, messageText, suggestionMount };
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
    
    //Typewriter effect with stop functionality
    function typeText(element, rawText, speed = 20, onDone = () => {}) {
        const parsedText = marked.parse(rawText);
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = parsedText;

        const textContent = tempDiv.innerText;  // what we animate
        const finalHTML   = tempDiv.innerHTML;  // what we'd show if not stopped

        let index = 0;
        let timerId = null;
        let stopped = false;   // <- NEW: indicates user hit Stop
        let finished = false;  // <- internal guard

        function cleanup() {
            if (timerId) clearTimeout(timerId);
            timerId = null;
        }

        function stopNow() {
            if (finished) return;
            stopped = true;
            cleanup();
            // leave whatever is already typed as-is (no finalHTML swap)
            activeTypeController = null;
            updateSendButton();
            // Intentionally DO NOT call onDone() here (no suggestions, etc.)
        }

        function step() {
            if (stopped || finished) return;
            if (index < textContent.length) {
                element.innerText = textContent.substring(0, index + 1);
                index++;
                scrollToBottom();
                timerId = setTimeout(step, speed);
            } else {
                // Completed naturally — render final HTML for formatting
                finished = true;
                cleanup();
                // Only if not stopped: commit final HTML + links
                element.innerHTML = finalHTML;
                attachLinkHandlers(element);
                scrollToBottom();
                activeTypeController = null;
                updateSendButton();
                onDone();  // suggestions, etc.
            }
        }

        // start
        step();

        // return controller to caller
        return {
            finishNow: stopNow
        };
    }

    // Stop/Send click handler (unchanged logic, but now finishNow truncates)
    sendButton.addEventListener('click', () => {
        if (activeTypeController) {
            // User wants to cut the reply where it is
            activeTypeController.finishNow();
        } else {
            sendMessage(false);
        }
    });

    // NEW: render follow-up suggestions under a specific bot message
    function renderSuggestions(mountEl, suggestions) {
        if (!mountEl) return;
        mountEl.innerHTML = "";
        if (!Array.isArray(suggestions) || suggestions.length === 0) return;

        suggestions.forEach((q) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'suggestion-chip';
            chip.textContent = q;
            chip.addEventListener('click', () => {
                userInput.value = q;
                sendMessage(true);
            });
            mountEl.appendChild(chip);
        });
        scrollToBottom();
    }

    function sendMessage(fromSuggestion = false) {
        const message = userInput.value.trim();
        if (!message) return;

        appendMessage(message, 'user');

        userInput.value = "";
        scrollToBottom();

        // Show Typing Indicator (for the "thinking" phase)
        const { messageDiv: typingDiv } = appendMessage("", 'bot', true);

        fetch(CHAT_ENDPOINT, {
            method: 'POST',
            body: JSON.stringify({ user_message: message }),
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.bot_reply || "Error");
                });
            }
            return response.json();
        })
        .then(data => {
            setTimeout(() => {
                chatContainer.removeChild(typingDiv);

                // Create actual bot message and start typing animation
                const { messageText, suggestionMount } = appendMessage("", 'bot');

                // Start typing and capture controller
                activeTypeController = typeText(
                    messageText,
                    data.bot_reply || "",
                    20,
                    () => {
                        // When typing completes, render suggestions (and ensure button resets)
                        renderSuggestions(suggestionMount, data.suggested_questions || []);
                    }
                );

                // Switch button to STOP while typing
                updateSendButton();

                // Render suggestions only after typing is done (also happens if user hits Stop)
                // We also render suggestions once more inside onDone above.
                // No-op here; handled in onDone.
            }, 600);
        })
        .catch(error => {
            // If typing indicator still exists, try to remove it safely
            try { chatContainer.removeChild(typingDiv); } catch (e) {}
            appendMessage("Error: " + error.message, 'bot');
            console.error('Error:', error);
            // Ensure button is back to send
            activeTypeController = null;
            updateSendButton();
        });
    }

    // Single click handler: Stop if typing, otherwise Send
    sendButton.addEventListener('click', () => {
        if (activeTypeController) {
            activeTypeController.finishNow();
        } else {
            sendMessage(false);
        }
    });

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !activeTypeController) {
            sendMessage(false);
        }
    });

    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: "smooth"
        });
    }
});

// Keep your unload → logout
window.addEventListener('beforeunload', function() {
    navigator.sendBeacon('/logout');
});