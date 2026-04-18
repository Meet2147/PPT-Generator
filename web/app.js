const pricingGrid = document.getElementById("pricing-grid");
const billingToggle = document.getElementById("billing-toggle");
const deckForm = document.getElementById("deck-form");
const result = document.getElementById("result");
const designSelect = document.getElementById("design_number");
const buyerNameInput = document.getElementById("buyer_name");
const buyerEmailInput = document.getElementById("buyer_email");

let activePlan = "monthly";
let catalog = { plans: [], designs: [] };

async function boot() {
  const response = await fetch("/api/v1/catalog");
  catalog = await response.json();
  renderDesigns();
  renderPricing();
}

function renderDesigns() {
  designSelect.innerHTML = catalog.designs
    .map(
      (design) =>
        `<option value="${design.id}">${design.name} • ${design.best_for}</option>`
    )
    .join("");
}

function renderPricing() {
  pricingGrid.innerHTML = catalog.plans
    .map((plan) => {
      const price = activePlan === "monthly" ? plan.price_monthly : plan.price_annual;
      const suffix = activePlan === "monthly" ? "/mo" : "/yr";
      return `
        <article class="pricing-card ${plan.recommended ? "recommended" : ""}">
          <div>
            <p class="eyebrow">${plan.audience}</p>
            <strong>${plan.name}</strong>
            <p>${plan.description}</p>
          </div>
          <div class="plan-price">$${price}<small>${suffix}</small></div>
          <div class="plan-savings">${plan.savings_label}</div>
          <ul>${plan.features.map((feature) => `<li>${feature.label}</li>`).join("")}</ul>
          <button class="primary-button subscribe-button" data-tier="${plan.slug}" type="button">${plan.cta}</button>
        </article>
      `;
    })
    .join("");
}

billingToggle?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-plan]");
  if (!button) {
    return;
  }
  activePlan = button.dataset.plan;
  for (const toggle of billingToggle.querySelectorAll(".toggle-button")) {
    toggle.classList.toggle("active", toggle === button);
  }
  renderPricing();
});

deckForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.hidden = false;
  result.innerHTML = "Generating your deck...";

  const payload = {
    topic: document.getElementById("topic").value,
    objective: document.getElementById("objective").value,
    audience: document.getElementById("audience").value,
    tone: document.getElementById("tone").value,
    design_number: Number(document.getElementById("design_number").value),
    slide_count: Number(document.getElementById("slide_count").value),
    plan: activePlan,
    include_images: true,
  };

  try {
    const response = await fetch("/api/v1/presentations/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to generate deck.");
    }

    result.innerHTML = `
      <strong>${data.title}</strong><br />
      ${data.summary}<br /><br />
      <a href="${data.download_url}">Download ${data.filename}</a><br />
      ${data.fallback_used ? "Fallback narrative used because no model response was available." : "AI narrative generated successfully."}
    `;
  } catch (error) {
    result.textContent = error.message;
  }
});

pricingGrid?.addEventListener("click", async (event) => {
  const button = event.target.closest(".subscribe-button");
  if (!button) {
    return;
  }

  try {
    const response = await fetch("/api/v1/billing/checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tier: button.dataset.tier,
        billing_cycle: activePlan,
        name: buyerNameInput?.value || "",
        email: buyerEmailInput?.value || "",
        quantity: 1,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to create subscription checkout.");
    }
    if (!window.Razorpay) {
      throw new Error("Razorpay Checkout failed to load.");
    }

    const checkout = new window.Razorpay({
      key: data.key_id,
      subscription_id: data.subscription_id,
      name: "DeckMint",
      description: data.description,
      prefill: data.customer,
      notes: {
        tier: button.dataset.tier,
        billing_cycle: activePlan,
      },
      theme: { color: "#ff7a29" },
      handler() {
        window.alert("Subscription authorized. Final status will sync through Razorpay webhooks.");
      },
    });
    checkout.open();
  } catch (error) {
    window.alert(error.message);
  }
});

boot();
