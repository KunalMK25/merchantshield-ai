// The currency symbol follows the data being served (rupees for the synthetic data, dollars for Sparkov).
// The server reports it in /health; App sets it once, and every page formats amounts through here.
let symbol = "₹";

export const setCurrency = (s) => {
  if (s) symbol = s;
};

export const currencySymbol = () => symbol;

export function money(value) {
  return typeof value === "number" ? `${symbol}${Math.round(value).toLocaleString("en-IN")}` : "—";
}

// Short form for charts. Lakh ("L") is an Indian convention, so it is used for rupees only.
export function compactMoney(amount) {
  if (symbol === "₹" && amount >= 100000) return `${symbol}${(amount / 100000).toFixed(1)}L`;
  if (amount >= 1000000) return `${symbol}${(amount / 1000000).toFixed(1)}M`;
  if (amount >= 1000) return `${symbol}${(amount / 1000).toFixed(1)}K`;
  return `${symbol}${amount}`;
}
