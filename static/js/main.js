console.log("SignBridge AI loaded");


const quickItems = document.querySelectorAll(".quick-item");


quickItems.forEach(item => {

    item.addEventListener("click", () => {

        const word = item.querySelector("strong").textContent;

        document.querySelector(".sentence").textContent = word;

        console.log("Selected:", word);

    });

});


const soundButton =
    document.querySelector(".sound-button");


if (soundButton) {

    soundButton.addEventListener("click", () => {

        const text =
            document.querySelector(".sentence").textContent;

        if ("speechSynthesis" in window) {

            const speech =
                new SpeechSynthesisUtterance(text);

            speech.lang = "en-US";

            window.speechSynthesis.speak(speech);

        }

    });

}