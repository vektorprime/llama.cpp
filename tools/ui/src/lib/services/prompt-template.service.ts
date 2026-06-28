import { API_PROMPT_TEMPLATES } from '$lib/constants/api-endpoints';
import { getJsonHeaders } from '$lib/utils/api-headers';
import type { ApiPromptTemplateMeta, ApiPromptTemplateSaveResponse } from '$lib/types/api';

export class PromptTemplateService {
	static async listTemplates(): Promise<ApiPromptTemplateMeta[]> {
		const headers = getJsonHeaders();
		const response = await fetch(API_PROMPT_TEMPLATES.LIST, { headers });
		if (!response.ok) {
			throw new Error(`Failed to list templates: ${response.status} ${response.statusText}`);
		}
		const data = await response.json();
		return data.templates || [];
	}

	static async saveTemplate(messages: Array<{ role: string; content: string }>, name?: string, folder?: string): Promise<ApiPromptTemplateSaveResponse> {
		const headers = getJsonHeaders();
		const body: Record<string, unknown> = { messages };
		if (name) body.name = name;
		if (folder) body.folder = folder;
		const response = await fetch(API_PROMPT_TEMPLATES.SAVE, {
			method: 'POST',
			headers,
			body: JSON.stringify(body)
		});
		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error((errorData as { error?: { message?: string } })?.error?.message || `Save failed: ${response.status}`);
		}
		return response.json();
	}

	static async deleteTemplate(id: string): Promise<void> {
		const headers = getJsonHeaders();
		const response = await fetch(API_PROMPT_TEMPLATES.DELETE(id), {
			method: 'DELETE',
			headers
		});
		if (!response.ok) {
			throw new Error(`Failed to delete template: ${response.status} ${response.statusText}`);
		}
	}

	static async getTemplate(id: string): Promise<ApiPromptTemplateMeta> {
		const headers = getJsonHeaders();
		const response = await fetch(API_PROMPT_TEMPLATES.LIST + '/' + id, { headers });
		if (!response.ok) {
			throw new Error(`Failed to get template: ${response.status} ${response.statusText}`);
		}
		return response.json();
	}

	static async updateTemplate(id: string, updates: { name?: string; folder?: string }): Promise<ApiPromptTemplateMeta> {
		const headers = getJsonHeaders();
		const response = await fetch(API_PROMPT_TEMPLATES.LIST + '/' + id, {
			method: 'POST',
			headers,
			body: JSON.stringify(updates)
		});
		if (!response.ok) {
			throw new Error(`Failed to update template: ${response.status} ${response.statusText}`);
		}
		return response.json();
	}
}
