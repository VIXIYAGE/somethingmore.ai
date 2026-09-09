const API_URL = "https://somethingmore-ai-pichhla-khatam.onrender.com";

const fileInput = document.getElementById("fileInput");
const fileTitle = document.getElementById("fileTitle");
const output = document.getElementById("output");
const resultPanel = document.getElementById("resultPanel");

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        fileTitle.textContent = file.name;
    }
});

async function askModel() {
    const file = fileInput.files[0];
    const question = document.getElementById("question").value;

    if (!file) {
        alert("Upload a JSON file first.");
        return;
    }
    if (!question.trim()) {
        alert("Enter a question first.");
        return;
    }

    try {
        output.textContent =
            "PROCESSING...\n\n" +
            "ENCODING -> TENSOR -> EMBEDDING -> SLM";
        resultPanel.style.display = "block";

        const text = await file.text();
        const json = JSON.parse(text);

        const response = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                data: json.data,
                question: question
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Backend error");
        }

        output.textContent =
            result.answer || JSON.stringify(result, null, 2);
    } catch (error) {
        output.textContent = "ERROR\n\n" + error.message;
    }
}