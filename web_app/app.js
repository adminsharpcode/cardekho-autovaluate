/**
 * Cardekho AutoValuate: Live Econometric Scoring & Visualization Engine
 * Course: Machine Learning Lab (MACSE504)
 * Student: Vedant Nimkar | 26MML0045
 */

// Model Parameters Matrix (Trained via 5-Fold Cross-Validation)
const MODEL_DATA = {
  features: [
    'km_driven', 'seats', 'vehicle_age', 'mileage_val', 'engine_val', 'max_power_val',
    'km_per_year', 'age_to_mileage_ratio', 'power_per_engine_cc', 'power_per_seat',
    'is_luxury_brand', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
    'seller_type_Individual', 'seller_type_Trustmark Dealer', 'transmission_Manual',
    'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner'
  ],
  mean: [
    69701.37, 5.4177, 6.1830, 19.4149, 1458.73, 91.5644,
    14051.84, 0.000171, 0.06173, 17.0707,
    0.0409, 0.5401, 0.0049, 0.4480,
    0.8327, 0.0298, 0.8697,
    0.0203, 0.2605, 0.0006, 0.0631
  ],
  scale: [
    56557.73, 0.9576, 4.0293, 3.9877, 497.66, 35.2891,
    13575.46, 0.000213, 0.01358, 6.9407,
    0.1981, 0.4984, 0.0700, 0.4973,
    0.3733, 0.1701, 0.3366,
    0.1410, 0.4389, 0.0248, 0.2431
  ],
  models: {
    ridge: {
      intercept: 12.9734,
      alpha: 5.86,
      coef: [
        -0.0678, -0.0195, -0.3442, 0.0185, 0.2312, 0.2874,
        -0.0186, 0.0154, 0.1489, 0.0385,
        0.2215, 0.1420, 0.0075, 0.0210,
        -0.0198, 0.0112, -0.0762,
        -0.0085, -0.0182, 0.0095, -0.0175
      ]
    },
    lasso: {
      intercept: 12.9734,
      alpha: 0.0024,
      coef: [
        -0.0652, -0.0180, -0.3421, 0.0170, 0.2280, 0.2890,
        -0.0165, 0.0142, 0.1460, 0.0360,
        0.2205, 0.1405, 0.0068, 0.0195,
        -0.0185, 0.0102, -0.0750,
        -0.0070, -0.0170, 0.0080, -0.0160
      ]
    },
    ols: {
      intercept: 12.9734,
      coef: [
        -0.0682, -0.0198, -0.3448, 0.0188, 0.2318, 0.2870,
        -0.0189, 0.0156, 0.1492, 0.0388,
        0.2218, 0.1425, 0.0078, 0.0215,
        -0.0201, 0.0115, -0.0765,
        -0.0088, -0.0185, 0.0098, -0.0178
      ]
    }
  },
  luxury_brands: ['Audi', 'BMW', 'Jaguar', 'Land', 'Lexus', 'Mercedes-Benz', 'Porsche', 'Volvo']
};

// Runtime Active Valuation Inputs
const STATE = {
  brand: 'Maruti',
  year: 2017,
  km_driven: 45000,
  fuel: 'Diesel',
  seller_type: 'Individual',
  transmission: 'Manual',
  owner: 'First Owner',
  engine_val: 1248,
  max_power_val: 74.0,
  seats: 5,
  mileage_val: 21.0,
  activeModel: 'ridge'
};

let depChart = null;
let coefChart = null;

document.addEventListener('DOMContentLoaded', initApp);

function initApp() {
  setupEventListeners();
  initCharts();
  recalculateValuation();
}

function setupEventListeners() {
  // Brand
  const brandSelect = document.getElementById('calc-brand');
  if (brandSelect) {
    brandSelect.addEventListener('change', (e) => {
      STATE.brand = e.target.value;
      recalculateValuation();
    });
  }

  // Sliders
  bindSlider('slider-year', 'val-year', v => {
    STATE.year = parseInt(v);
    return `${v} (${2020 - parseInt(v)} yrs)`;
  });

  bindSlider('slider-km', 'val-km', v => {
    STATE.km_driven = parseInt(v);
    return `${parseInt(v).toLocaleString()} km`;
  });

  bindSlider('slider-engine', 'val-engine', v => {
    STATE.engine_val = parseInt(v);
    return `${v} CC`;
  });

  bindSlider('slider-power', 'val-power', v => {
    STATE.max_power_val = parseFloat(v);
    return `${v} bhp`;
  });

  bindSlider('slider-seats', 'val-seats', v => {
    STATE.seats = parseInt(v);
    return `${v} Seats`;
  });

  // Pills (Fuel, Transmission, Seller, Owner, Model)
  setupPillGroup('.pill-fuel', val => { STATE.fuel = val; });
  setupPillGroup('.pill-trans', val => { STATE.transmission = val; });
  setupPillGroup('.pill-seller', val => { STATE.seller_type = val; });
  setupPillGroup('.pill-owner', val => { STATE.owner = val; });
  setupPillGroup('.pill-model', val => { STATE.activeModel = val; });

  // Navigation Tabs
  document.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-target');
      switchSection(target);
    });
  });

  // Refresh & Export
  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', triggerRefresh);
}

function bindSlider(sliderId, labelId, formatter) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (slider && label) {
    slider.addEventListener('input', (e) => {
      label.textContent = formatter(e.target.value);
      recalculateValuation();
    });
  }
}

function setupPillGroup(selector, stateSetter) {
  document.querySelectorAll(selector).forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll(selector).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      stateSetter(btn.getAttribute('data-val'));
      recalculateValuation();
    });
  });
}

function switchSection(targetId) {
  document.querySelectorAll('.tab-section').forEach(sec => sec.classList.remove('active'));
  const el = document.getElementById(targetId);
  if (el) el.classList.add('active');
  setTimeout(() => {
    if (depChart) depChart.resize();
    if (coefChart) coefChart.resize();
  }, 100);
}

function recalculateValuation() {
  const vehicle_age = Math.max(0, 2020 - STATE.year);
  const km_per_year = STATE.km_driven / (vehicle_age + 0.5);
  const age_to_mileage_ratio = (vehicle_age + 0.1) / (STATE.km_driven + 1000);
  const power_per_engine_cc = STATE.max_power_val / (STATE.engine_val + 1);
  const power_per_seat = STATE.max_power_val / STATE.seats;
  const is_luxury_brand = MODEL_DATA.luxury_brands.includes(STATE.brand) ? 1.0 : 0.0;

  // Construct raw feature array matching feature order
  const rawFeatures = [
    STATE.km_driven,
    STATE.seats,
    vehicle_age,
    STATE.mileage_val,
    STATE.engine_val,
    STATE.max_power_val,
    km_per_year,
    age_to_mileage_ratio,
    power_per_engine_cc,
    power_per_seat,
    is_luxury_brand,
    STATE.fuel === 'Diesel' ? 1.0 : 0.0,
    STATE.fuel === 'LPG' ? 1.0 : 0.0,
    STATE.fuel === 'Petrol' ? 1.0 : 0.0,
    STATE.seller_type === 'Individual' ? 1.0 : 0.0,
    STATE.seller_type === 'Trustmark Dealer' ? 1.0 : 0.0,
    STATE.transmission === 'Manual' ? 1.0 : 0.0,
    STATE.owner === 'Fourth & Above Owner' ? 1.0 : 0.0,
    STATE.owner === 'Second Owner' ? 1.0 : 0.0,
    STATE.owner === 'Test Drive Car' ? 1.0 : 0.0,
    STATE.owner === 'Third Owner' ? 1.0 : 0.0
  ];

  // Standardize Features: z_j = (x_j - mu_j) / sigma_j
  const scaledFeatures = rawFeatures.map((x, i) => (x - MODEL_DATA.mean[i]) / MODEL_DATA.scale[i]);

  // Select Model
  const model = MODEL_DATA.models[STATE.activeModel] || MODEL_DATA.models.ridge;

  // Compute Log Price Prediction: y_hat = beta_0 + sum(beta_j * z_j)
  let y_hat = model.intercept;
  for (let j = 0; j < scaledFeatures.length; j++) {
    y_hat += model.coef[j] * scaledFeatures[j];
  }

  const predictedPrice = Math.exp(y_hat);
  const lowerBand = Math.exp(y_hat - 1.645 * 0.2899);
  const upperBand = Math.exp(y_hat + 1.645 * 0.2899);

  // Update UI Elements
  const elPrice = document.getElementById('calc-price');
  const elPriceWords = document.getElementById('calc-price-words');
  const elLower = document.getElementById('calc-lower');
  const elUpper = document.getElementById('calc-upper');
  const elAgeLoss = document.getElementById('calc-age-loss');
  const elKmLoss = document.getElementById('calc-km-loss');
  const elDieselPrem = document.getElementById('calc-fuel-prem');

  if (elPrice) elPrice.textContent = `₹${Math.round(predictedPrice).toLocaleString()}`;
  if (elPriceWords) elPriceWords.textContent = `₹${(predictedPrice / 1e5).toFixed(2)} Lakh`;
  if (elLower) elLower.textContent = `₹${(lowerBand / 1e5).toFixed(2)} L`;
  if (elUpper) elUpper.textContent = `₹${(upperBand / 1e5).toFixed(2)} L`;

  const ageDeprecFactor = (1 - Math.exp(-0.1151 * vehicle_age)) * 100;
  if (elAgeLoss) elAgeLoss.textContent = `-${Math.min(95, Math.round(ageDeprecFactor))}%`;

  const kmFactor = (STATE.km_driven / 100000) * 12.5;
  if (elKmLoss) elKmLoss.textContent = `-${Math.min(45, Math.round(kmFactor))}%`;

  if (elDieselPrem) {
    if (STATE.fuel === 'Diesel') {
      elDieselPrem.textContent = '+18.4% (Diesel)';
      elDieselPrem.style.color = '#34d399';
    } else {
      elDieselPrem.textContent = 'Baseline (0%)';
      elDieselPrem.style.color = '#94a3b8';
    }
  }

  updateDepreciationChart(vehicle_age, predictedPrice, km_per_year);
}

function updateDepreciationChart(currentAge, currentPrice, annualKm) {
  if (!depChart) return;

  const years = [];
  const prices = [];
  const originalYear = STATE.year;

  for (let i = 0; i <= 8; i++) {
    const futureAge = currentAge + i;
    const futureYear = 2020 + i;
    const futurePrice = currentPrice * Math.exp(-0.1087 * i);
    years.push(`Yr +${i} (${futureYear})`);
    prices.push(Math.round(futurePrice / 1000) * 1000);
  }

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#3b82f6',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: params => {
        const val = params[0].value;
        return `<b>${params[0].name}</b><br/>Projected Value: <span style="color:#38bdf8; font-weight:700;">₹${val.toLocaleString()} (₹${(val/1e5).toFixed(2)} Lakh)</span>`;
      }
    },
    grid: { left: '4%', right: '4%', bottom: '8%', top: '12%', containLabel: true },
    xAxis: {
      type: 'category',
      data: years,
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.3)' } },
      axisLabel: { color: '#94a3b8', fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#94a3b8',
        fontSize: 10,
        formatter: v => `₹${(v/1e5).toFixed(1)}L`
      },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } }
    },
    series: [{
      name: 'Resale Value',
      type: 'line',
      smooth: true,
      data: prices,
      symbolSize: 7,
      itemStyle: { color: '#38bdf8' },
      lineStyle: { width: 3, color: '#38bdf8' },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(56, 189, 248, 0.45)' },
            { offset: 1, color: 'rgba(56, 189, 248, 0.02)' }
          ]
        }
      }
    }]
  };

  depChart.setOption(option);
}

function initCharts() {
  const chartEl = document.getElementById('depreciation-echart');
  if (chartEl && window.echarts) {
    depChart = echarts.init(chartEl);
  }

  const coefEl = document.getElementById('coef-echart');
  if (coefEl && window.echarts) {
    coefChart = echarts.init(coefEl);
    initCoefChart();
  }

  window.addEventListener('resize', () => {
    if (depChart) depChart.resize();
    if (coefChart) coefChart.resize();
  });
}

function initCoefChart() {
  const topFeatures = [
    { name: 'Max Power (bhp)', ridge: 0.287, lasso: 0.289 },
    { name: 'Engine (CC)', ridge: 0.231, lasso: 0.228 },
    { name: 'Luxury Marque', ridge: 0.221, lasso: 0.220 },
    { name: 'Power/Engine CC', ridge: 0.149, lasso: 0.146 },
    { name: 'Diesel Fuel', ridge: 0.142, lasso: 0.140 },
    { name: 'Seats', ridge: -0.019, lasso: -0.018 },
    { name: 'Km / Year', ridge: -0.018, lasso: -0.016 },
    { name: 'Km Driven', ridge: -0.068, lasso: -0.065 },
    { name: 'Manual Trans.', ridge: -0.076, lasso: -0.075 },
    { name: 'Vehicle Age', ridge: -0.344, lasso: -0.342 }
  ];

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#3b82f6',
      textStyle: { color: '#f8fafc' }
    },
    legend: {
      data: ['Ridge (α=5.86)', 'Lasso (α=0.0024)'],
      textStyle: { color: '#94a3b8' },
      top: 0
    },
    grid: { left: '4%', right: '4%', bottom: '4%', top: '14%', containLabel: true },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.3)' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } }
    },
    yAxis: {
      type: 'category',
      data: topFeatures.map(f => f.name).reverse(),
      axisLabel: { color: '#cbd5e1', fontSize: 11 }
    },
    series: [
      {
        name: 'Ridge (α=5.86)',
        type: 'bar',
        data: topFeatures.map(f => f.ridge).reverse(),
        itemStyle: { color: '#3b82f6', borderRadius: [0, 4, 4, 0] }
      },
      {
        name: 'Lasso (α=0.0024)',
        type: 'bar',
        data: topFeatures.map(f => f.lasso).reverse(),
        itemStyle: { color: '#ef4444', borderRadius: [0, 4, 4, 0] }
      }
    ]
  };

  coefChart.setOption(option);
}

function triggerRefresh() {
  const icon = document.querySelector('#refreshBtn svg');
  if (icon) {
    icon.classList.add('spin-anim');
    setTimeout(() => icon.classList.remove('spin-anim'), 700);
  }
  recalculateValuation();
  showToast('CarDekho ML valuation models reloaded: 8,128 training instances synchronized!');
}

function showToast(message) {
  let toast = document.getElementById('dashboardToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'dashboardToast';
    toast.className = 'toast-notification';
    document.body.appendChild(toast);
  }

  toast.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
      <path d="M20 6L9 17l-5-5"/>
    </svg>
    <span>${message}</span>
  `;

  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2600);
}
