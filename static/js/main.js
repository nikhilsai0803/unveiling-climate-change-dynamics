/* ═══════════════════════════════════════════════
   Unveiling Climate Change Dynamics Through Earth Surface Temperature Analysis — main.js v4
═══════════════════════════════════════════════ */

// ── Animated Gradient Background Canvas ──
(function() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, t = 0;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const orbs = [
    { x: 0.15, y: 0.2,  r: 0.35, color: [56, 189, 248],  speed: 0.0007, ox: 0.08, oy: 0.06 },
    { x: 0.85, y: 0.75, r: 0.28, color: [52, 211, 153],  speed: 0.0009, ox: 0.06, oy: 0.09 },
    { x: 0.5,  y: 0.5,  r: 0.22, color: [167, 139, 250], speed: 0.0006, ox: 0.12, oy: 0.05 },
    { x: 0.8,  y: 0.15, r: 0.18, color: [251, 191, 36],  speed: 0.001,  ox: 0.07, oy: 0.1  },
  ];

  function draw() {
    t++;
    ctx.clearRect(0, 0, w, h);
    orbs.forEach((o, i) => {
      const cx = (o.x + Math.sin(t * o.speed + i) * o.ox) * w;
      const cy = (o.y + Math.cos(t * o.speed * 1.3 + i) * o.oy) * h;
      const r  = o.r * Math.min(w, h);
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grd.addColorStop(0, `rgba(${o.color.join(',')},0.07)`);
      grd.addColorStop(1, `rgba(${o.color.join(',')},0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Navbar scroll effect ──
(function() {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  function onScroll() {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

// ── Hamburger mobile menu ──
(function() {
  const btn     = document.getElementById('hamburger');
  const overlay = document.getElementById('mobileOverlay');
  if (!btn || !overlay) return;

  btn.addEventListener('click', () => {
    overlay.classList.toggle('open');
    document.body.style.overflow = overlay.classList.contains('open') ? 'hidden' : '';
  });

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      overlay.classList.remove('open');
      document.body.style.overflow = '';
    }
  });

  // close on link click
  overlay.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      overlay.classList.remove('open');
      document.body.style.overflow = '';
    });
  });
})();

// ── Scroll reveal ──
(function() {
  const els = document.querySelectorAll('.scroll-reveal');
  if (!els.length) return;

  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  els.forEach(el => io.observe(el));
})();

// ── Active nav link ──
(function() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href');
    if (href === path || (href !== '/' && path.startsWith(href))) {
      a.classList.add('active');
    }
  });
})();

// ── Contact form (client-side simulation) ──
(function() {
  const form   = document.getElementById('contactForm');
  const banner = document.getElementById('successBanner');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const btn = form.querySelector('.submit-btn');
    btn.textContent = 'Sending…';
    btn.disabled = true;
    setTimeout(() => {
      if (banner) { banner.style.display = 'block'; }
      form.reset();
      btn.textContent = 'Message Sent!';
      btn.style.background = 'linear-gradient(135deg,#34d399,#10b981)';
      setTimeout(() => {
        btn.textContent = 'Send Message';
        btn.disabled = false;
        btn.style.background = '';
        if (banner) banner.style.display = 'none';
      }, 5000);
    }, 1200);
  });
})();
