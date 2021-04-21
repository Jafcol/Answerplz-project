function _sl(s, c) {
	return (c || document).querySelector(s);
}

(function() {
	var html = document.querySelector('html');
	
	if (/idcacG/.test(html.className))
		return;
	
	html.className += ' idcacG';
	
	var c = 0, i = setInterval(function() {
		
		// General privacy reminder
		var e1 = _sl('form[action^="/signin/privacyreminder"] > div > span > div:not([role]) > div:not([tabindex]) span + div');
		if (e1) e1.click();
		
		// google.fr/flights
		var e2 = _sl('#gb[role="banner"] > div > div[style^="behavior"] > div > span + a[role="button"] + a[role="button"]');
		if (e2) e2.click();
		
		// #cns=1
		if (document.location.hash == '#cns=1')
			document.location.hash = '#cns=0';
		
		c++;
		
		if (c == 300)
			clearInterval(i);
	}, 500);
})();