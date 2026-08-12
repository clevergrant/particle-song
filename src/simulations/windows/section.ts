/**
 * Collapsible-section helpers shared by every window builder.
 */

export interface Section {
	readonly section: HTMLElement
	readonly body: HTMLElement
}

export function makeSection(title: string, defaultOpen: boolean): Section {
	const section = document.createElement("div")
	section.className = "settings-section" + (defaultOpen ? " open" : "")
	section.dataset.section = title

	const header = document.createElement("div")
	header.className = "settings-section-header"
	header.textContent = title
	header.addEventListener("click", () => section.classList.toggle("open"))
	section.appendChild(header)

	const body = document.createElement("div")
	body.className = "settings-section-body"
	section.appendChild(body)

	return { section, body }
}

export function getOpenSections(container: HTMLElement): Set<string> {
	const open = new Set<string>()
	for (const el of container.querySelectorAll(".settings-section.open")) {
		const name = (el as HTMLElement).dataset.section
		if (name) open.add(name)
	}
	return open
}

export function restoreOpenSections(container: HTMLElement, open: Set<string>) {
	for (const el of container.querySelectorAll(".settings-section")) {
		const name = (el as HTMLElement).dataset.section
		if (name && open.has(name)) (el as HTMLElement).classList.add("open")
	}
}
