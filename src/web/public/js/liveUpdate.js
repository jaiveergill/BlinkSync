setInterval(() => {
    const xhttp = new XMLHttpRequest();
    xhttp.open("GET", "data.txt", true);
    // xhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
    xhttp.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            // Response
            var response = this.responseText.split(",")
            console.log(response)
            document.querySelector("#redmorse").textContent = response[0]
            document.querySelector("#redletter").textContent = response[1]
            document.querySelector("#text").textContent = response[2] + response[3]
        }
    };
    xhttp.send();
}, 50)