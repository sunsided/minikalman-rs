// Starts every experiment on the page; one failing does not stop the others.

import { themeToggle } from "./lab.js";

themeToggle(document.getElementById("theme"));

const experiments = ["basics", "radar", "outliers", "numerics", "smoothing", "parameters", "attitude"];

for (const name of experiments) {
  const section = document.getElementById(name);
  if (!section) continue;
  import(`./exp-${name}.js`)
    .then((module) => module.start(section))
    .catch((error) => {
      console.error(`${name}:`, error);
      const note = document.createElement("p");
      note.className = "fallback";
      note.textContent = `This experiment could not start: ${error.message}. It needs a browser with WebAssembly.`;
      section.querySelector(".instrument")?.prepend(note);
    });
}
