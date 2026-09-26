(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const two = (value) => Number(value || 0).toFixed(2);
  const openModal = (selector) => { const node = $(selector); if (node) node.hidden = false; };
  const closeModal = (selector) => { const node = $(selector); if (node) node.hidden = true; };
  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-close-modal]');
    if (button) closeModal(button.dataset.closeModal);
  });
  const SEV = {fire: 4, heat_no_flame: 3, needs_review: 2, rgb_only: 2, flame_no_heat: 1, clear: 0};
  const api = async (url, options = {}) => { const response = await fetch(url, {headers: options.body instanceof FormData ? {} : {'Content-Type':'application/json'}, ...options}); const data = await response.json().catch(() => ({})); if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`); return data; };
  const toast = (message, kind = '') => { const node=document.createElement('div'); node.className=`toast ${kind ? `toast--${kind}` : ''}`; node.textContent=message; document.querySelector('#toasts')?.append(node); setTimeout(()=>node.remove(),4000); };
  window.FireWatch={$, $$, two, openModal, closeModal, api, toast, onState:null,poll:async()=>{try{const data=await api('/api/state');$("#systemText").textContent=data.armed?'SYSTEM ARMED':'SYSTEM STANDBY';window.FireWatch.onState?.(data);window.dispatchEvent(new CustomEvent('fw:state',{detail:data}))}catch(_){$("#systemText").textContent='SERVER OFFLINE'}}};
  const clock=()=>{const node=document.querySelector('#clock');if(node)node.textContent=new Date().toLocaleTimeString([], {hour12:false})}; clock();setInterval(clock,1000);window.FireWatch.poll();setInterval(window.FireWatch.poll,2000);
})();
