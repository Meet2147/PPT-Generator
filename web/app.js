const pricingGrid = document.getElementById("pricing-grid");
const billingToggle = document.getElementById("billing-toggle");
const deckForm = document.getElementById("deck-form");
const result = document.getElementById("result");
const designSelect = document.getElementById("design_number");
const buyerNameInput = document.getElementById("buyer_name");
const buyerEmailInput = document.getElementById("buyer_email");
const previewGrid = document.getElementById("preview-grid");
const previewSummary = document.getElementById("preview-summary");
const historyList = document.getElementById("history-list");
const userChip = document.getElementById("user-chip");
const exportPdfInput = document.getElementById("export_pdf");
const authForm = document.getElementById("auth-form");
const authResult = document.getElementById("auth-result");
const authSubmit = document.getElementById("auth-submit");
const signupNameWrap = document.getElementById("signup-name-wrap");
const authName = document.getElementById("auth-name");
const authEmail = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const showLogin = document.getElementById("show-login");
const showSignup = document.getElementById("show-signup");
const previewSubmit = document.getElementById("preview-submit");
const appConfig = window.DECKMINT_CONFIG || {};
const apiBaseUrl = (appConfig.apiBaseUrl || window.location.origin).replace(/\/$/, "");

let activePlan = "monthly";
let authMode = "login";
let catalog = { plans: [], designs: [] };
let session = loadSession();

boot();

async function boot() {
  bindAuthMode();
  renderSession();
  await loadCatalog();
  if (session?.token) {
    await refreshHistory();
  }
}

async function loadCatalog() {
  const response = await fetch(`${apiBaseUrl}/api/v1/catalog`);
  catalog = await response.json();
  renderDesigns();
  renderPricing();
}

function bindAuthMode() {
  showLogin?.addEventListener("click", () => setAuthMode("login"));
  showSignup?.addEventListener("click", () => setAuthMode("signup"));

  authForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    authResult.hidden = false;
    authResult.textContent = authMode === "login" ? "Signing you in..." : "Creating your account...";

    const endpoint = authMode === "login" ? "/api/v1/auth/login" : "/api/v1/auth/signup";
    const payload = {
      email: authEmail.value,
      password: authPassword.value,
    };
    if (authMode === "signup") {
      payload.name = authName.value;
    }

    try {
      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Authentication failed.");
      }
      session = { token: data.access_token, user: data.user };
      persistSession(session);
      renderSession();
      authResult.textContent = `${authMode === "login" ? "Logged in" : "Account created"} as ${data.user.name}.`;
      authPassword.value = "";
      await refreshHistory();
    } catch (error) {
      authResult.textContent = error.message;
    }
  });

  previewSubmit?.addEventListener("click", async () => {
    await previewDeck();
  });

  deckForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await generateDeck();
  });
}

function setAuthMode(mode) {
  authMode = mode;
  authSubmit.textContent = mode === "login" ? "Login" : "Create account";
  signupNameWrap.hidden = mode !== "signup";
  showLogin.classList.toggle("active", mode === "login");
  showSignup.classList.toggle("active", mode === "signup");
}

function renderSession() {
  if (session?.user) {
    userChip.hidden = false;
    userChip.textContent = `Signed in as ${session.user.name}`;
    buyerNameInput.value = session.user.name;
    buyerEmailInput.value = session.user.email;
  } else {
    userChip.hidden = true;
  }
}

function renderDesigns() {
  designSelect.innerHTML = catalog.designs
    .map((design) => `<option value="${design.id}">${design.name} • ${design.best_for}</option>`)
    .join("");
}

function renderPricing() {
  pricingGrid.innerHTML = catalog.plans
    .map((plan) => {
      const price = activePlan === "monthly" ? plan.price_monthly : plan.price_annual;
      const suffix = activePlan === "monthly" ? "/mo" : "/yr";
      const formattedPrice = new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 0,
      }).format(price);
      return `
        <article class="pricing-card ${plan.recommended ? "recommended" : ""}">
          <div>
            <p class="eyebrow">${plan.audience}</p>
            <strong>${plan.name}</strong>
            <p>${plan.description}</p>
          </div>
          <div class="plan-price">${formattedPrice}<small>${suffix}</small></div>
          <div class="plan-quota">Up to ${plan.monthly_presentation_cap} presentations per month</div>
          <div class="plan-savings">${plan.savings_label}</div>
          <ul>${plan.features.map((feature) => `<li>${feature.label}</li>`).join("")}</ul>
          <button class="primary-button subscribe-button" data-tier="${plan.slug}" type="button">${plan.cta}</button>
        </article>
      `;
    })
    .join("");
  renderLifetimeOffer();
}

function renderLifetimeOffer() {
  const container = document.getElementById("lifetime-offer");
  if (!container || !catalog.lifetime_offer) {
    return;
  }

  const offer = catalog.lifetime_offer;
  const price = formatInr(offer.price);
  const original = formatInr(offer.original_price);
  container.innerHTML = `
    <article class="pricing-card lifetime-card">
      <div>
        <p class="eyebrow">${offer.badge}</p>
        <strong>${offer.name}</strong>
        <p>${offer.description}</p>
      </div>
      <div class="lifetime-meter">
        <div class="lifetime-meter-bar">
          <span style="width:${Math.min((offer.claimed_spots / 1000) * 100, 100)}%"></span>
        </div>
        <div class="lifetime-meter-copy">
          <strong>${offer.remaining_spots}</strong> spots left out of 1000
        </div>
      </div>
      <div class="lifetime-price-row">
        <div class="plan-price">${price}<small>one time</small></div>
        <div class="lifetime-original">${original}</div>
      </div>
      <div class="plan-savings">${offer.note}</div>
      <ul>${offer.features.map((feature) => `<li>${feature.label}</li>`).join("")}</ul>
      <button class="primary-button subscribe-button" data-tier="${offer.slug}" data-billing-cycle="lifetime" type="button" ${offer.sold_out ? "disabled" : ""}>${offer.sold_out ? "Sold out" : offer.cta}</button>
    </article>
  `;
}

billingToggle?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-plan]");
  if (!button) return;
  activePlan = button.dataset.plan;
  for (const toggle of billingToggle.querySelectorAll(".toggle-button")) {
    toggle.classList.toggle("active", toggle === button);
  }
  renderPricing();
});

pricingGrid?.addEventListener("click", handlePricingClick);
document.getElementById("lifetime-offer")?.addEventListener("click", handlePricingClick);

async function handlePricingClick(event) {
  const button = event.target.closest(".subscribe-button");
  if (!button) return;
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/billing/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tier: button.dataset.tier,
        billing_cycle: button.dataset.billingCycle || activePlan,
        name: buyerNameInput?.value || "",
        email: buyerEmailInput?.value || "",
        quantity: 1,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to create checkout.");

    if (data.flow_type === "payment_link" && data.checkout_url) {
      window.location.href = data.checkout_url;
      return;
    }

    if (!window.Razorpay) throw new Error("Razorpay Checkout failed to load.");
    const checkout = new window.Razorpay({
      key: data.key_id,
      subscription_id: data.subscription_id,
      name: "DeckMint",
      description: data.description,
      prefill: data.customer,
      notes: {
        tier: button.dataset.tier,
        billing_cycle: button.dataset.billingCycle || activePlan,
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
}

async function previewDeck() {
  if (!ensureAuth()) return;
  previewSummary.textContent = "Generating preview...";
  previewGrid.innerHTML = "";
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/presentations/preview`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(generationPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to preview slides.");
    previewSummary.textContent = data.summary;
    previewGrid.innerHTML = data.slides
      .map(
        (slide) => `
          <article class="slide-preview" style="--slide-accent:${slide.accent};">
            <div class="slide-preview-header">
              <span>Slide ${String(slide.number).padStart(2, "0")}</span>
              <span>${slide.vibe}</span>
            </div>
            <h4>${slide.title}</h4>
            <p>${slide.content}</p>
          </article>
        `
      )
      .join("");
  } catch (error) {
    previewSummary.textContent = error.message;
  }
}

async function generateDeck() {
  if (!ensureAuth()) return;
  result.hidden = false;
  result.innerHTML = "Generating files...";
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/presentations/generate`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(generationPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to generate deck.");
    result.innerHTML = `
      <strong>${data.title}</strong><br />
      ${data.summary}<br /><br />
      <a href="${data.download_url}">Download ${data.filename}</a>
      ${data.pdf_download_url ? `<br /><a href="${data.pdf_download_url}">Download ${data.pdf_filename}</a>` : ""}
    `;
    autoDownload(data.download_url);
    if (data.pdf_download_url) autoDownload(data.pdf_download_url);
    await refreshHistory();
  } catch (error) {
    result.textContent = error.message;
  }
}

async function refreshHistory() {
  if (!session?.token) return;
  const response = await fetch(`${apiBaseUrl}/api/v1/decks/history`, {
    headers: authHeaders(),
  });
  const data = await response.json();
  if (!response.ok) return;
  historyList.innerHTML = data.length
    ? data
        .map(
          (item) => `
            <article class="history-card">
              <div>
                <strong>${item.title}</strong>
                <p>${item.topic} • ${item.audience} • ${item.slide_count} slides</p>
              </div>
              <div class="history-links">
                <a href="${item.pptx_download_url}">PPTX</a>
                ${item.pdf_download_url ? `<a href="${item.pdf_download_url}">PDF</a>` : ""}
              </div>
            </article>
          `
        )
        .join("")
    : `<div class="history-empty">No decks yet. Generate your first presentation to see it here.</div>`;
}

function generationPayload() {
  return {
    topic: document.getElementById("topic").value,
    objective: document.getElementById("objective").value,
    audience: document.getElementById("audience").value,
    tone: document.getElementById("tone").value,
    design_number: Number(document.getElementById("design_number").value),
    slide_count: Number(document.getElementById("slide_count").value),
    plan: activePlan,
    include_images: true,
    export_pdf: exportPdfInput.checked,
  };
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session.token}`,
  };
}

function ensureAuth() {
  if (session?.token) return true;
  authResult.hidden = false;
  authResult.textContent = "Please sign in or create an account first.";
  return false;
}

function persistSession(value) {
  localStorage.setItem("deckmint_session", JSON.stringify(value));
}

function loadSession() {
  try {
    return JSON.parse(localStorage.getItem("deckmint_session") || "null");
  } catch {
    return null;
  }
}

function formatInr(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function autoDownload(url) {
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}
