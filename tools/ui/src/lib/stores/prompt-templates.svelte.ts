import { PromptTemplateService } from '$lib/services/prompt-template.service';
import type { ApiPromptTemplateMeta, ApiPromptTemplateSaveResponse } from '$lib/types/api';

class PromptTemplatesStore {
	private _templates = $state<ApiPromptTemplateMeta[]>([]);
	private _loading = $state(false);
	private _error = $state<string | null>(null);

	get templates() { return this._templates; }
	get loading() { return this._loading; }
	get error() { return this._error; }

	async fetchTemplates() {
		this._loading = true;
		this._error = null;
		try {
			this._templates = await PromptTemplateService.listTemplates();
		} catch (e: unknown) {
			this._error = e instanceof Error ? e.message : 'Failed to load templates';
		} finally {
			this._loading = false;
		}
	}

	async saveTemplate(messages: Array<{ role: string; content: string }>, name?: string, folder?: string): Promise<ApiPromptTemplateSaveResponse> {
		const result = await PromptTemplateService.saveTemplate(messages, name, folder);
		await this.fetchTemplates();
		return result;
	}

	async updateTemplate(id: string, updates: { name?: string; folder?: string }) {
		await PromptTemplateService.updateTemplate(id, updates);
		await this.fetchTemplates();
	}

	async deleteTemplate(id: string) {
		await PromptTemplateService.deleteTemplate(id);
		await this.fetchTemplates();
	}

	async getTemplate(id: string): Promise<ApiPromptTemplateMeta> {
		return PromptTemplateService.getTemplate(id);
	}
}

export const promptTemplatesStore = new PromptTemplatesStore();
