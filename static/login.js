document.addEventListener('DOMContentLoaded', () => {
    const loginButton = document.getElementById('loginButton');

    loginButton.addEventListener('click', () => {
        console.log("Login button clicked!"); // Debugging
        handleLogin();
    });

    document.getElementById('password').addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            console.log("Enter key pressed!");
            handleLogin();
        }
    });
});

function handleLogin() {
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
    const errorMessage = document.getElementById('error-message');

    if (username === 'construction' && password === 'password') {
        document.getElementById('loginModal').style.display = 'none';
        document.body.classList.remove('blur-background');

        // Enable chat input and button
        document.getElementById('user-input').disabled = false;
        document.getElementById('send-button').disabled = false;

        console.log("Login successful!");
    } else {
        errorMessage.textContent = "Invalid username or password!";
        console.log("Login failed!");
    }
}