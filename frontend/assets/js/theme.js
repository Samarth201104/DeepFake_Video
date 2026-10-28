(() => {
  const button = document.getElementById("themeToggle");
  try {
    if (localStorage.getItem("framecheck-theme") === "light") document.documentElement.classList.add("light");
  } catch {}
  button.addEventListener("click", () => {
    document.documentElement.classList.toggle("light");
    const light = document.documentElement.classList.contains("light");
    try { localStorage.setItem("framecheck-theme", light ? "light" : "dark"); } catch {}
    button.setAttribute("aria-label", `Switch to ${light ? "dark" : "light"} theme`);
  });
  const navigation = document.getElementById("navigation"), menu = document.getElementById("menuToggle");
  menu.addEventListener("click", () => {
    const open = navigation.classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
  });
  navigation.addEventListener("click", (event) => {
    if (!event.target.closest("a")) return;
    navigation.classList.remove("open");
    menu.setAttribute("aria-expanded", "false");
  });
})();
