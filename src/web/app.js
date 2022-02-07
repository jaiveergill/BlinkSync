const express = require("express");
const app = express();
const path = require('path')
const fs = require('fs')
const db_url = process.env.DB_URL
const Text = require('./schemas/text')
// // // const mongoose = require('mongoose')
// // // mongoose
// // //     .connect(db_url, {
// // //         useNewUrlParser: true,
// // //         useUnifiedTopology: true,
// // //     })
// // //     .then(() => {
// // //         console.log("Connection open!");
// // //     })
// // //     .catch((err) => {
// // //         console.log("Error in connecting to MongoDB:", err);
// // //     });

// // async function initText() {
// //     const text = new Text({
// //         text: "Blink here for text to appear!",
// //         main: true,
// //     })
// //     await text.save()
// // }
// initText()
app.use(express.urlencoded({
    extended: true
}));
app.use(express.json());
app.use(express.static("public"));
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

app.get('/', async (req, res) => {
    res.render('home.ejs')
})

app.get('/data.txt', (req, res) => {
    fs.readFile('data.txt', (err, data) => {
        res.send(data)
    })
})

app.post('/data', (req, res) => {
    const {
        text
    } = req.body
    fs.writeFile('data.txt', text, err => {
        console.log("Error writing to file data.txt: ", err)
    })
    res.send("OK")
})
const port = process.env.PORT || 3000
app.listen(port, () => {
    console.log("Listening on port 3000")
})