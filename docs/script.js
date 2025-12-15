// Navigation functionality for OCS Camera documentation

document.addEventListener("DOMContentLoaded", () => {
  // Set active navigation link based on current page
  setActiveNavLink();

  // Section highlighting removed - no highlight functionality

  // Smooth scroll for anchor links
  setupSmoothScrolling();

  // Mobile menu toggle (if needed in future)
  setupMobileMenu();
});

const setActiveNavLink = () => {
  const currentPage = window.location.pathname.split("/").pop();
  const navLinks = document.querySelectorAll(".top-nav .nav-links a");

  navLinks.forEach((link) => {
    link.classList.remove("active");
    const href = link.getAttribute("href");

    if ((currentPage === "" || currentPage === "index.html") && href === "index.html") {
      link.classList.add("active");
    } else if (currentPage === "docs.html" && href === "docs.html") {
      link.classList.add("active");
    } else if (currentPage === "report.html" && href === "report.html") {
      link.classList.add("active");
    }
  });
};

// Section highlighting removed - no highlight functionality

const setupSmoothScrolling = () => {
  const links = document.querySelectorAll('a[href^="#"]');

  links.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();

      const targetId = e.target.getAttribute("href").substring(1);
      const targetElement = document.getElementById(targetId);

      if (targetElement) {
        const offsetTop = targetElement.offsetTop - 80; // Account for fixed header
        window.scrollTo({
          top: offsetTop,
          behavior: "smooth",
        });
      }
    });
  });
};

const setupMobileMenu = () => {
  // Placeholder for future mobile menu functionality
  // This can be expanded if a mobile hamburger menu is needed
};

// Add keyboard navigation support
document.addEventListener("keydown", (e) => {
  // Press 'Escape' to return to top
  if (e.key === "Escape") {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // Press '/' to focus search (if search is added in future)
  if (e.key === "/" && !e.ctrlKey && !e.metaKey) {
    const searchInput = document.querySelector("#search-input");
    if (searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
  }
});

// Print-friendly adjustments
window.addEventListener("beforeprint", () => {
  document.body.classList.add("printing");
});

window.addEventListener("afterprint", () => {
  document.body.classList.remove("printing");
});
