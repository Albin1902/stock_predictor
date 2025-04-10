
console.log("JavaScript file loaded.");

const tbody = document.querySelector("#data-table tbody");
const resultDiv = document.getElementById("result");
const errorDiv = document.getElementById("error");
const loadingDiv = document.getElementById("loading");

function generateRowData() {
  const open = (Math.random() * 100 + 100).toFixed(2);
  const high = (parseFloat(open) + Math.random() * 5).toFixed(2);
  const low = (parseFloat(open) - Math.random() * 5).toFixed(2);
  const close = ((parseFloat(high) + parseFloat(low)) / 2).toFixed(2);
  const volume = Math.floor(Math.random() * 50000000 + 10000000);
  return [open, high, low, close, volume];
}

function fillTable() {
  tbody.innerHTML = "";
  for (let i = 0; i < 60; i++) {
    const row = document.createElement("tr");
    const [open, high, low, close, volume] = generateRowData();
    row.innerHTML = `
      <td>${i + 1}</td>
      <td><input type="number" class="form-control" value="${open}"></td>
      <td><input type="number" class="form-control" value="${high}"></td>
      <td><input type="number" class="form-control" value="${low}"></td>
      <td><input type="number" class="form-control" value="${close}"></td>
      <td><input type="number" class="form-control" value="${volume.toLocaleString()}"></td>
    `;
    tbody.appendChild(row);
  }
}

document.getElementById("predictBtn").onclick = async () => {
  resultDiv.textContent = "";
  errorDiv.textContent = "";
  loadingDiv.textContent = "⏳ Predicting...";
  loadingDiv.style.display = "block";

  const rows = document.querySelectorAll("#data-table tbody tr");
  const sequence = [];

  for (let row of rows) {
    const inputs = row.querySelectorAll("input");
    const values = Array.from(inputs).map(input => parseFloat(input.value.replace(/,/g, '')));
    if (values.some(isNaN)) {
      errorDiv.textContent = "❗ Please fill in all 60 rows correctly.";
      loadingDiv.style.display = "none";
      return;
    }
    sequence.push(values);
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence })
    });

    const data = await response.json();
    if (response.ok) {
      resultDiv.innerHTML = `<b>📊 Predicted Closing Price:</b> $${data.predicted_price}`;
    } else {
      errorDiv.textContent = `❌ Error: ${data.error}`;
    }
  } catch (err) {
    errorDiv.textContent = "❌ Could not connect to backend. Is Flask running?";
  } finally {
    loadingDiv.style.display = "none";
  }
};

document.getElementById("resetBtn").onclick = fillTable;

document.getElementById("fetchBtn").onclick = async () => {
  const symbol = document.getElementById("stockSymbol").value.trim();
  if (!symbol) {
    errorDiv.textContent = "Please enter a stock ticker.";
    return;
  }

  resultDiv.textContent = "";
  errorDiv.textContent = "";
  loadingDiv.textContent = "📡 Fetching stock data...";
  loadingDiv.style.display = "block";

  try {
    const res = await fetch("http://127.0.0.1:5000/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: symbol })
    });
    const data = await res.json();

    if (res.ok) {
      tbody.innerHTML = "";
      data.data.forEach((row, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${i + 1}</td>
          ${row.map((val, index) => {
            const formatted = index === 5 ? parseInt(val).toLocaleString() : parseFloat(val).toFixed(2);
            return `<td><input type="text" class="form-control" value="${formatted}"></td>`;
          }).join('')}
        `;
        tbody.appendChild(tr);
      });
    } else {
      errorDiv.textContent = data.error;
    }
  } catch (err) {
    errorDiv.textContent = "❌ Error fetching stock data.";
  } finally {
    loadingDiv.style.display = "none";
  }
};

fillTable();
