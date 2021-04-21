// Vars

var cached_rules = {},
	whitelisted_domains = {},
	tab_list = {};


// Common functions

function getHostname(url, cleanup)
{
	try
	{
		if (url.indexOf('http') != 0)
			throw true;
		
		var a = new URL(url);
		
		return (typeof cleanup == 'undefined' ? a.hostname : a.hostname.replace(/^w{2,3}\d*\./i, ''));
	}
	catch(error)
	{
		return false;
	}
}


// Whitelisting

function updateWhitelist()
{
	chrome.storage.local.get('whitelisted_domains', function(r) {
		if (typeof r.whitelisted_domains != 'undefined')
			whitelisted_domains = r.whitelisted_domains;
	});
}

updateWhitelist();

chrome.runtime.onMessage.addListener(function(request, info){
	if (request == 'update_whitelist')
		updateWhitelist();
});

function isWhitelisted(tab)
{
	if (typeof whitelisted_domains[tab.hostname] != 'undefined')
		return true;
	
	for (var i in tab.host_levels)
		if (typeof whitelisted_domains[tab.host_levels[i]] != 'undefined')
			return true;
	
	return false;
}

function getWhitelistedDomain(tab)
{
	if (typeof whitelisted_domains[tab.hostname] != 'undefined')
		return tab.hostname;
	
	for (var i in tab.host_levels)
		if (typeof whitelisted_domains[tab.host_levels[i]] != 'undefined')
			return tab.host_levels[i];
	
	return false;
}

function toggleWhitelist(tab)
{
	if (tab.url.indexOf('http') != 0 || !tab_list[tab.id])
		return;
	
	if (tab_list[tab.id].whitelisted)
	{
		var hostname = getWhitelistedDomain(tab_list[tab.id]);
		delete whitelisted_domains[tab_list[tab.id].hostname];
	}
	else
		whitelisted_domains[tab_list[tab.id].hostname] = true;
	
	chrome.storage.local.set({'whitelisted_domains': whitelisted_domains}, function(){
		for (var i in tab_list)
			if (tab_list[i].hostname == tab_list[tab.id].hostname)
				tab_list[i].whitelisted = !tab_list[tab.id].whitelisted;
	});
}


// Maintain tab list

function getPreparedTab(tab)
{
	tab.hostname = false;
	tab.whitelisted = false;
	tab.host_levels = [];
	
	if (tab.url)
	{
		tab.hostname = getHostname(tab.url, true);
		
		if (tab.hostname)
		{
			var parts = tab.hostname.split('.');
			
			for (var i=parts.length; i>=2; i--)
				tab.host_levels.push(parts.slice(-1*i).join('.'));
			
			tab.whitelisted = isWhitelisted(tab);
		}
	}
	
	return tab;
}

function onCreatedListener(tab)
{
    tab_list[tab.id] = getPreparedTab(tab);
}

function onUpdatedListener(tabId, changeInfo, tab) {
	if (changeInfo.status)
		tab_list[tab.id] = getPreparedTab(tab);
}

function onRemovedListener(tabId) {
    if (tab_list[tabId])
		delete tab_list[tabId];
}

function recreateTabList()
{
	tab_list = {};
	
	chrome.tabs.query({}, function(results) {
		results.forEach(onCreatedListener);
		
		for (var i in tab_list)
			doTheMagic(tab_list[i].id);
	});
}

chrome.tabs.onCreated.addListener(onCreatedListener);
chrome.tabs.onUpdated.addListener(onUpdatedListener);
chrome.tabs.onRemoved.addListener(onRemovedListener);

chrome.runtime.onStartup.addListener(function(d){
	cached_rules = {};
	recreateTabList();
});

chrome.runtime.onInstalled.addListener(function(d){
	cached_rules = {};
	
	if (d.reason == "update" && chrome.runtime.getManifest().version > d.previousVersion)
		recreateTabList();
});


// URL blocking

function blockUrlCallback(d)
{
	if (tab_list[d.tabId] && !tab_list[d.tabId].whitelisted && d.url && d.url.indexOf(',') == -1)
	{
		var clean_url = d.url.split('?')[0];
		
		for (var i in block_urls.common)
		{
			if ((block_urls.common[i].q && d.url.indexOf(block_urls.common[i].r) > -1) || (!block_urls.common[i].q && clean_url.indexOf(block_urls.common[i].r) > -1))
			{
				// Check for exceptions
				if (block_urls.common[i].e && tab_list[d.tabId].host_levels.length > 0)
					for (var level in tab_list[d.tabId].host_levels)
						for (var exception in block_urls.common[i].e)
							if (block_urls.common[i].e[exception] == tab_list[d.tabId].host_levels[level])
								return {cancel:false};
				
				return {cancel:true};
			}
		}
		
		if (d.tabId > -1 && tab_list[d.tabId].host_levels.length > 0)
		{
			for (var level in tab_list[d.tabId].host_levels)
			{
				if (block_urls.specific[tab_list[d.tabId].host_levels[level]])
				{
					var rules = block_urls.specific[tab_list[d.tabId].host_levels[level]];
					
					for (var i in rules)
						if ((rules[i].q && d.url.indexOf(rules[i].r) > -1) || (!rules[i].q && clean_url.indexOf(rules[i].r) > -1))
							return {cancel:true};
				}
			}
		}
	}
	
	return {cancel:false};
}

chrome.webRequest.onBeforeRequest.addListener(blockUrlCallback, {urls:["http://*/*", "https://*/*"], types:["script","stylesheet","xmlhttprequest"]}, ["blocking"]);


// Reporting

function reportWebsite(info, tab)
{
	if (tab.url.indexOf('http') != 0 || !tab_list[tab.id])
		return;
	
	
	var hostname = getHostname(tab.url);
	
	if (hostname.length == 0)
		return;
	
	
	if (tab_list[tab.id].whitelisted)
	{
		return chrome.notifications.create('report', {
			type: "basic",
			title: chrome.i18n.getMessage("reportSkippedTitle", hostname),
			message: chrome.i18n.getMessage("reportSkippedMessage"),
			iconUrl: "icons/48.png"
		});
	}
	
	
	chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu/report/"+chrome.runtime.getManifest().version+'/'+encodeURIComponent(encodeURIComponent(tab.url))});
}


// Adding custom CSS/JS

function activateDomain(hostname, tabId, frameId)
{
	if (!cached_rules[hostname])
		cached_rules[hostname] = rules[hostname] || {};
	
	if (!cached_rules[hostname])
		return false;
	
	var r = cached_rules[hostname];
	
	if (typeof r.s != 'undefined')
		chrome.tabs.insertCSS(tabId, {code: r.s, frameId: frameId, runAt: 'document_start'});
	else if (typeof r.c != 'undefined')
		chrome.tabs.insertCSS(tabId, {code: commons[r.c], frameId: frameId, runAt: 'document_start'});
	else if (typeof r.j != 'undefined')
		chrome.tabs.executeScript(tabId, {file: 'data/js/'+(r.j > 0 ? 'common'+r.j : hostname)+'.js', frameId: frameId, runAt: 'document_end'});
	else
		return false;
	
	return true;
}

function doTheMagic(tabId, frameId)
{
	if (!tab_list[tabId] || tab_list[tabId].url.indexOf('http') != 0)
		return;
	
	if (tab_list[tabId].whitelisted)
		return;
	
	// Common CSS rules
	chrome.tabs.insertCSS(tabId, {file: "data/css/common.css", frameId: frameId || 0, runAt: 'document_start'});
	
	// Common social embeds
	chrome.tabs.executeScript(tabId, {file:'data/js/embeds.js', frameId: frameId || 0, runAt: 'document_end'});
	
	if (activateDomain(tab_list[tabId].hostname, tabId, frameId || 0))
		return;
	
	for (var level in tab_list[tabId].host_levels)
		if (activateDomain(tab_list[tabId].host_levels[level], tabId, frameId || 0))
			return true;
	
	// Common JS rules when custom rules don't exist
	chrome.tabs.executeScript(tabId, {file: 'data/js/common.js', frameId: frameId || 0, runAt: 'document_end'});
}

chrome.webNavigation.onCommitted.addListener(function(tab) {
	if (tab.frameId > 0)
		return;
	
	tab_list[tab.tabId] = getPreparedTab(tab);
	
	doTheMagic(tab.tabId);
});

chrome.webRequest.onResponseStarted.addListener(function(tab) {
	if (tab.frameId > 0)
		doTheMagic(tab.tabId, tab.frameId);
}, {urls: ['<all_urls>'], types: ["sub_frame", "script"]});


// Update notification

chrome.runtime.onInstalled.addListener(function(d){
	if (d.reason == "update" && chrome.runtime.getManifest().version > d.previousVersion)
	{
// 		chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu/whats-new/2019/?b=c"});
		
// 		chrome.notifications.create('update', {
// 			type: "basic",
// 			title: "I don't care about cookies - updated!",
// 			message: "Happy with it? You can help. Ask your favorite website to link to i-dont-care-about-cookies.eu from its cookie warning.",
// 			iconUrl: "icons/48.png"/*,
// 			buttons:[{title: chrome.i18n.getMessage("menuSupport")}]*/
// 		});
// 		
// 		// chrome.notifications.onButtonClicked.addListener(function(){
// 		//	chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu/"});
// 		// });
	}
	
	if (d.reason == "install") {
		chrome.storage.local.get('is_installed', function(r) {
			if (typeof r.is_installed == 'undefined') {
				chrome.storage.local.set({'is_installed': true}, function() {
// 					chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu"});
				});
			}
		});
	}
});


// Toolbar menu

chrome.runtime.onMessage.addListener(function(request, info, sendResponse) {
	if (typeof request == 'object')
	{
		if (request.tabId && tab_list[request.tabId])
		{
			if (request.command == 'get_active_tab')
			{
				var response = {tab: tab_list[request.tabId]};
				
				if (response.tab.whitelisted)
					response.tab.hostname = getWhitelistedDomain(tab_list[request.tabId]);
				
				sendResponse(response);
			}
			else if (request.command == 'toggle_extension')
				toggleWhitelist(tab_list[request.tabId]);
			else if (request.command == 'report_website')
				chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu/report/"+chrome.runtime.getManifest().version+'/'+encodeURIComponent(encodeURIComponent(tab_list[request.tabId].url))});
			else if (request.command == 'refresh_page')
				chrome.tabs.executeScript(request.tabId, {code:'window.location.reload();'});
		}
		else
		{
			if (request.command == 'open_support_page')
				chrome.tabs.create({url:"https://www.i-dont-care-about-cookies.eu/"});
			else if (request.command == 'open_options_page')
				chrome.tabs.create({url:chrome.runtime.getURL('data/options.html')});
			else if (request.command == 'open_ecosia_page')
				chrome.tabs.create({url:'http://bit.ly/2NCamFa'});
		}
	}
});