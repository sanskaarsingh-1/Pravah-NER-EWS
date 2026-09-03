/* PRAVAH — Site-wide Google Translate Widget logic
   Includes a searchable language picker. */
(function () {
  function toggleWidget() {
    var w = document.getElementById('pravahTranslateWidget');
    if (w) w.classList.toggle('active');
  }
  window.pravahToggleTranslate = toggleWidget;

  document.addEventListener('click', function (e) {
    var widget = document.getElementById('pravahTranslateWidget');
    var btn = document.getElementById('pravahTranslateBtn');
    if (widget && btn && !widget.contains(e.target) && !btn.contains(e.target)) {
      widget.classList.remove('active');
    }
  });

  function addLanguageSearch() {
    var widget = document.getElementById('pravahTranslateWidget');
    if (!widget || widget.querySelector('.pravah-language-search')) return false;

    var select = widget.querySelector('.goog-te-combo');
    if (!select || !select.options.length) return false;

    var search = document.createElement('input');
    search.type = 'search';
    search.className = 'pravah-language-search';
    search.placeholder = 'Search language...';
    search.setAttribute('aria-label', 'Search language');
    search.autocomplete = 'off';

    widget.insertBefore(search, select);

    var options = Array.prototype.slice.call(select.options);

    search.addEventListener('input', function () {
      var query = search.value.trim().toLowerCase();
      var visibleCount = 0;

      options.forEach(function (option) {
        var text = (option.textContent || option.innerText || '').toLowerCase();
        var matches = !query || text.indexOf(query) !== -1;
        option.hidden = !matches;
        if (matches) visibleCount++;
      });

      select.setAttribute(
        'aria-label',
        visibleCount ? visibleCount + ' matching languages' : 'No matching languages'
      );
    });

    search.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        search.value = '';
        search.dispatchEvent(new Event('input'));
        search.blur();
      }
    });

    return true;
  }

  function watchForGoogleLanguageSelect() {
    if (addLanguageSearch()) return;

    var widget = document.getElementById('pravahTranslateWidget');
    if (!widget) return;

    var observer = new MutationObserver(function () {
      if (addLanguageSearch()) observer.disconnect();
    });
    observer.observe(widget, { childList: true, subtree: true });

    setTimeout(function () {
      observer.disconnect();
    }, 15000);
  }

  // Called by the Google Translate script once it loads.
  window.googleTranslateElementInit = function () {
    /* global google */
    new google.translate.TranslateElement(
      { pageLanguage: 'en', autoDisplay: false },
      'pravahTranslateWidget'
    );
    watchForGoogleLanguageSelect();
  };
})();
