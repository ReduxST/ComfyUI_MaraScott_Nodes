import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

if (!window.marascott) {
	window.marascott = {}
}
if (!window.marascott.McBoaty_TilePrompter_Ollama_v1) {
	window.marascott.McBoaty_TilePrompter_Ollama_v1 = {
		init: false,
		clean: false,
        params: {
            denoise: {
                values: [],
            }
        },
        message: {
            prompts: [],
            tiles: [],
            denoises: [],
        },
        inputs: {
            prompts: [],
            tiles: [],
            denoises: [],
        },
	}
}

function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

function clearProcessedFlag() {
    document.querySelectorAll('textarea[data-processed]').forEach(textarea => {
        textarea.removeAttribute('data-processed');
    });
}

export const McBoatyWidgets = {

    validateDenoiseValue: (value) => {
        if (value === '') {
            return true;
        }
        // Try to convert the value to a float
        let num = parseFloat(value);
        // Check if the conversion was successful and the number is between 0.00 and 1.00
        return !isNaN(num) && num >= 0.00 && num <= 1.00;
    },

    formatDenoiseValue: (value) => {

        if (value === '') {
            return value;
        }

        // Convert the value to a string to check and modify
        let strValue = value.toString();
        
        // If the string starts with a decimal point, prepend a zero
        if (strValue.startsWith('.')) {
            strValue = '0' + strValue;
        }
        
        // Convert the string back to a float
        return parseFloat(strValue).toFixed(2);
    },

    WRAPPER: (key, index, prompt, tile, denoise, node) => {
        console.log('[McBoaty] Creating tile', index, 'for node:', node?.id);
        const inputEl = document.createElement("div");
        inputEl.className = "comfy-wrapper-mcboaty";
        
        const wrapper = document.createElement("div");
        wrapper.style.height = "100%";
        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "10px";

        const text = document.createElement("p");
        text.textContent = String(index + 1).padStart(2, '0');
        
        const textarea = document.createElement("textarea");
        textarea.style.opacity = 0.6;
        textarea.style.flexGrow = 1;
        textarea.style.height = "100%";
        textarea.className = "comfy-multiline-input";
        textarea.value = prompt || "";
        textarea.placeholder = "tile "+text.textContent;
        textarea.dataset.index = index;
        textarea.dataset.nodeId = node.id;

        textarea.addEventListener('focusout', async function() {
            const index = parseInt(this.dataset.index);
            const nodeId = this.dataset.nodeId;
            
            console.log('[McBoaty] Validation - raw dataset:', {
                index: this.dataset.index,
                nodeId: this.dataset.nodeId
            });
            
            if (isNaN(index) || index < 0 || !nodeId) {
                console.error('[McBoaty] Invalid parameters:', {index, nodeId});
                return;
            }
            
            console.log('[McBoaty] Focusout event triggered on tile', index);
            this.value = this.value.trim();
            
            if(window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts[index] != this.value) {
                console.log('[McBoaty] Detected prompt change for tile', index, 'New value:', this.value);
                
                // Create URL with proper encoding
                const params = new URLSearchParams();
                params.append('index', String(index));
                params.append('prompt', this.value);
                params.append('node', String(nodeId));
                if (api.clientId) params.append('clientId', api.clientId);
                
                try {
                    const response = await fetch(`/MaraScott/McBoaty/Ollama/v1/set_prompt`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: params.toString()
                    });
                    const data = await response.json();
                    console.log('[McBoaty] Save response:', data);
                    
                    const requeueWidget = MaraScottMcBoatyOllamaNodeWidget.getByName(node, 'requeue');
                    if (requeueWidget) {
                        MaraScottMcBoatyOllamaNodeWidget.setValue(node, 'requeue', requeueWidget.value + 1);
                    }
                } catch (error) {
                    console.error('[McBoaty] Save failed:', error);
                }
            }
        });

        const input = document.createElement("input");
        input.style.opacity = 0.6;
        input.style.height = "100%";
        input.style.maxWidth = "1.8rem";
        input.style.flexShrink = "0";
        input.className = "comfy-multiline-input";
        input.value = denoise || '';
        input.placeholder = "denoise "+text.textContent;
        input.addEventListener('focusout', async function() {
            this.value = this.value.trim()
            if (! McBoatyWidgets.validateDenoiseValue(this.value)) {
                this.value = '';
            } else {
                this.value = McBoatyWidgets.formatDenoiseValue(this.value);
            }
            if(window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises[index] != this.value) {
                window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises[index] = this.value;
                const res = await (await fetch(`/MaraScott/McBoaty/Ollama/v1/set_denoise?index=${index}&denoise=${this.value}&node=${this.dataNodeId}&clientId=${api.clientId}`)).json();
                const nodeWidget = MaraScottMcBoatyOllamaNodeWidget.getByName(node, 'requeue');
                MaraScottMcBoatyOllamaNodeWidget.setValue(node, 'requeue', ++nodeWidget.value);
            }
        });
        
        var img = document.createElement('img');
        img.src = imageDataToUrl(tile);  // Replace with the actual image path
        img.alt = prompt;
        img.style.height = "100%";
        img.style.maxWidth = "128px";
        img.style.maxHeight = "128px";
        img.style.flexShrink = "0";
        wrapper.appendChild(text);
        wrapper.appendChild(img);
        wrapper.appendChild(textarea);
        wrapper.appendChild(input);
        inputEl.appendChild(wrapper);
        
        const widget = node.addDOMWidget(name, "customtext", inputEl, {
            getValue() {
                return inputEl.value;
            },
            setValue(v) {
                inputEl.value = v;
            },
        });
        widget.inputEl = inputEl;
        MaraScottMcBoatyOllamaNodeWidget.setValue(node, widget.name, prompt);
        
        textarea.addEventListener("input", () => {
            widget.callback?.(widget.value);
        });
    
        return widget;
    }
}

class MaraScottMcBoatyOllamaNodePrompter {

	static async clean(node) {

		window.marascott.McBoaty_TilePrompter_Ollama_v1.clean = false

        const cleanedLabel = " ... cleaned"
        const nodeTitle = node.title
		node.title = nodeTitle + cleanedLabel
        
        const res_prompts = await (await fetch(`/MaraScott/McBoaty/Ollama/v1/get_input_prompts?node=${node.id}`)).json();
        const res_denoises = await (await fetch(`/MaraScott/McBoaty/Ollama/v1/get_input_denoises?node=${node.id}`)).json();

        window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts = window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.prompts = res_prompts.prompts_in;
        window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.denoises = window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises = res_denoises.denoises_in;
        window.marascott.McBoaty_TilePrompter_Ollama_v1.message.tiles = window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.tiles;
        MaraScottMcBoatyOllamaNodeWidget.setValue(node, MaraScottMcBoatyOllamaNodeWidget.INDEX.name, MaraScottMcBoatyOllamaNodeWidget.INDEX.default);
        MaraScottMcBoatyOllamaNodeWidget.setValue(node, MaraScottMcBoatyOllamaNodeWidget.PROMPT.name, MaraScottMcBoatyOllamaNodeWidget.PROMPT.default);
        MaraScottMcBoatyOllamaNodeWidget.setValue(node, MaraScottMcBoatyOllamaNodeWidget.DENOISE.name, MaraScottMcBoatyOllamaNodeWidget.DENOISE.default);
        MaraScottMcBoatyOllamaNodeWidget.setValue(node, 'requeue', 0);

        node.widgets = node.widgets.filter(widget => {
            const focusOutEvent = new Event('focusout');
            if (widget.type == "customtext") {
                const textarea = widget.inputEl.querySelector('[placeholder^="tile "]');
                if ( textarea != null) {
                    const dataId = textarea.getAttribute('placeholder');
                    const indexValue = parseInt(dataId.replace('tile ', ''), 10);
                    const realIndexValue = indexValue - 1;
                    const value = window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.prompts[realIndexValue];
                    MaraScottMcBoatyOllamaNodeWidget.setValue(node, widget.name, value);
                    textarea.value = value;
                    textarea.dispatchEvent(focusOutEvent);
                }
            }
            return true;

        });

        MaraScottMcBoatyOllamaNodeWidget.refresh(node);
		setTimeout(() => {
			// Remove " (cleaned)" from the title
			node.title = nodeTitle;
		}, 500);

	}

}

class MaraScottMcBoatyOllamaNodeWidget {

	static INDEX = {
		name: "Filter by Indexes",
		default: "",
	}
	static PROMPT = {
		name: "Prompt",
		default: "",
	}
	static DENOISE = {
		name: "Denoise",
        values: ['unchanged', 'Use Global Denoise'],
		default: "unchanged",
		min: 0.00,
		max: 1.00,
		step: 0.01,
	}
	static CLEAN = {
		name: 'Reset',
		default: false,
	}

	static refresh(node) {
		node.widgets = node.widgets.filter(widget => {
			// Keep Ollama query and system prompt widgets
			if (widget.name === "ollama_query" || widget.name === "ollama_system_prompt") {
				return true;
			}
			// Remove all other customtext widgets and clean widget
			if (widget.type == "customtext" || widget.name === this.CLEAN.name) {
				widget.onRemove?.();
				return false;
			}
			return true;
		});

		this.setIndexInput(node)
		this.setPromptInput(node)
		this.setDenoiseInput(node)

		node.onResize?.(node.size);
		node.graph.setDirtyCanvas(true, true);

		this.setPrompterInputs(node)

		this.setCleanSwitch(node)

		node.onResize?.(node.size);
		node.graph.setDirtyCanvas(true, true);
	}

	static init(node) {

		this.setIndexInput(node)
		this.setPromptInput(node)
		this.setDenoiseInput(node)
		this.setCleanSwitch(node)

	}

	static getByName(node, name) {
		return node.widgets?.find((w) => w.name === name);
	}

	static setValue(node, name, value) {

		const nodeWidget = this.getByName(node, name);
		nodeWidget.value = value
		node.setProperty(name, nodeWidget.value ?? node.properties[name])
		node.setDirtyCanvas(true)

	}

	static setIndexInput(node) {

		const nodeWidget = this.getByName(node, this.INDEX.name);

		if (nodeWidget == undefined) {
			node.addWidget(
				"text",
				this.INDEX.name,
				this.INDEX.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.INDEX.name, value);
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
                    this.setValue(node, this.DENOISE.name, this.DENOISE.default);
                    this.refresh(node);
				},
				{}
			)
			this.setValue(node, this.INDEX.name, this.INDEX.default)
		}

	}

    static setPromptInput(node) {

		const nodeWidget = this.getByName(node, this.PROMPT.name);
        
		if (nodeWidget == undefined) {
            node.addWidget(
                "text",
				this.PROMPT.name,
				this.PROMPT.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    this.setValue(node, this.PROMPT.name, value);
                    
                    const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                    node.widgets = node.widgets.filter(widget => {

                        const focusOutEvent = new Event('focusout');
                        const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                        if (widget.type == "customtext") {
                            const textarea = widget.inputEl.querySelector('[placeholder^="tile "]');
                            if ( textarea != null) {
                                const dataId = textarea.getAttribute('placeholder');
                                const indexValue = parseInt(dataId.replace('tile ', ''), 10);
                                const realIndexValue = indexValue - 1;
                                const indexFound = index_filtered.indexOf(indexValue - 1);
                                if((input_list != "" && indexFound > -1) || input_list == "") {
                                    textarea.value = value;
                                    textarea.dispatchEvent(focusOutEvent);
                                }
                            }
                        }
                        return true;

                    });
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);

				},
				{}
			)
			this.setValue(node, this.PROMPT.name, this.PROMPT.default)
		}

	}

	static setDenoiseInput(node) {

		const nodeWidget = this.getByName(node, this.DENOISE.name);

		if (nodeWidget == undefined) {

			node.addWidget(
				"combo",
				this.DENOISE.name,
				this.DENOISE.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    this.setValue(node, this.DENOISE.name, value);
                    
                    if (value != "unchanged") {
                        if (value == 'Use Global Denoise') value = '';

                        const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                        node.widgets = node.widgets.filter(widget => {

                            const focusOutEvent = new Event('focusout');
                            const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                            if (widget.type == "customtext") {
                                const input = widget.inputEl.querySelector('[placeholder^="denoise "]');
                                if ( input != null) {
                                    const dataId = input.getAttribute('placeholder');
                                    const indexValue = parseInt(dataId.replace('denoise ', ''), 10);
                                    const realIndexValue = indexValue - 1;
                                    const indexFound = index_filtered.indexOf(indexValue - 1);
                                    if((input_list != "" && indexFound > -1) || input_list == "") {
                                        input.value = value;
                                        input.dispatchEvent(focusOutEvent);
                                    }
                                }
                            }
                            return true;

                        });
                        this.setValue(node, this.DENOISE.name, this.DENOISE.default);
    
                    }

				},
				{
					"values": window.marascott.McBoaty_TilePrompter_Ollama_v1.params.denoise.values
				}
			)
			this.setValue(node, this.DENOISE.name, this.DENOISE.default)
		}

	}


	static setCleanSwitch(node) {

		const nodeWidget = this.getByName(node, this.CLEAN.name);
		if (nodeWidget == undefined) {
			node.addWidget(
				"toggle",
				this.CLEAN.name,
				this.CLEAN.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.CLEAN.name, this.CLEAN.default)
                    MaraScottMcBoatyOllamaNodePrompter.clean(node)
				},
				{}
			)
			this.setValue(node, this.CLEAN.name, this.CLEAN.default)
		}

	}

    static setPrompterInputs(node) {

        let index_list = (node.properties[this.INDEX.name] ?? this.INDEX.default).trim().split(",");
        index_list = (index_list.length === 1 && index_list[0] === '') ? [] : index_list;
        const index_filtered = index_list.map(num => Number(num) - 1);

        for (const [index] of window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts.entries()) {
            if (index_list.length == 0 || index_filtered.indexOf(index) > -1) {
                const w = McBoatyWidgets.WRAPPER("tile "+index, index, window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts[index], window.marascott.McBoaty_TilePrompter_Ollama_v1.message.tiles[index], window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises[index], node);
            }
        }


    }

}

window.marascott.McBoaty_TilePrompter_Ollama_v1.params.denoise.values = MaraScottMcBoatyOllamaNodeWidget.DENOISE.values;
for (let i = MaraScottMcBoatyOllamaNodeWidget.DENOISE.min; i <= MaraScottMcBoatyOllamaNodeWidget.DENOISE.max; i = parseFloat((i + MaraScottMcBoatyOllamaNodeWidget.DENOISE.step).toFixed(2))) {
    window.marascott.McBoaty_TilePrompter_Ollama_v1.params.denoise.values.push(i.toFixed(2));
}


class McBoaty_TilePrompter_Ollama_v1 {
	constructor() {
		if (!window.__McBoaty_TilePrompter_Ollama_v1__) {
			window.__McBoaty_TilePrompter_Ollama_v1__ = Symbol("__McBoaty_TilePrompter_Ollama_v1__");
		}
		this.symbol = window.__McBoaty_TilePrompter_Ollama_v1__;
	}

	getState(node) {
		return node[this.symbol] || {};
	}

	setState(node, state) {
		node[this.symbol] = state;
		app.canvas.setDirty(true);
	}

	addStatusTagHandler(nodeType) {
		if (nodeType[this.symbol]?.statusTagHandler) {
			return;
		}
		if (!nodeType[this.symbol]) {
			nodeType[this.symbol] = {};
		}
		nodeType[this.symbol] = {
			statusTagHandler: true,
		};

		api.addEventListener("MaraScott/McBoaty/Ollama/v1/update_status", ({ detail }) => {
			let { node, progress, text } = detail;
			const n = app.graph.getNodeById(+(node || app.runningNodeId));
			if (!n) return;
			const state = this.getState(n);
			state.status = Object.assign(state.status || {}, { progress: text ? progress : null, text: text || null });
			this.setState(n, state);
		});

		const self = this;
		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			const r = onDrawForeground?.apply?.(this, arguments);
			const state = self.getState(this);
			if (!state?.status?.text) {
				return r;
			}

			const { fgColor, bgColor, text, progress, progressColor } = { ...state.status };

			ctx.save();
			ctx.font = "12px sans-serif";
			const sz = ctx.measureText(text);
			ctx.fillStyle = bgColor || "dodgerblue";
			ctx.beginPath();
			ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
			ctx.fill();

			if (progress) {
				ctx.fillStyle = progressColor || "green";
				ctx.beginPath();
				ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, (sz.width + 12) * progress, 20, 5);
				ctx.fill();
			}

			ctx.fillStyle = fgColor || "#fff";
			ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
			ctx.restore();
			return r;
		};
	}
}

const mcBoatyOllama = new McBoaty_TilePrompter_Ollama_v1();

app.registerExtension({
	name: "ComfyUI.MaraScott.McBoaty_TilePrompter_Ollama_v1",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

        mcBoatyOllama.addStatusTagHandler(nodeType);

		if (nodeData.name === "MaraScottMcBoatyTilePrompter_Ollama_v1") {

            const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				console.log("[McBoaty Ollama] Before execution:", this.id);
				const r = onExecuted?.apply?.(this, arguments);

				console.log("[McBoaty Ollama] Raw message:", message);
				
				window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.prompts = message.prompts_in || [];
				window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts = message.prompts_out || [];
				window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.denoises = message.denoises_in || [];
				window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises = message.denoises_out || [];
				window.marascott.McBoaty_TilePrompter_Ollama_v1.inputs.tiles = window.marascott.McBoaty_TilePrompter_Ollama_v1.message.tiles = message.tiles || [];

				console.log("[McBoaty Ollama] State after update:", {
					prompts: window.marascott.McBoaty_TilePrompter_Ollama_v1.message.prompts,
					tiles: window.marascott.McBoaty_TilePrompter_Ollama_v1.message.tiles,
					denoises: window.marascott.McBoaty_TilePrompter_Ollama_v1.message.denoises
				});

				MaraScottMcBoatyOllamaNodeWidget.refresh(this);
				console.log("[McBoaty Ollama] After refresh");

				return r;
			};
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.widgets = this.widgets.filter(widget => {
                    if (widget.name.startsWith("tile ")) {
                        widget.onRemove?.();
                        return false;
                    }
                    return true;
                });

                MaraScottMcBoatyOllamaNodeWidget.init(this);

                this.onResize?.(this.size);

                return r;
            }        
    
		} else {
			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = getExtraMenuOptions?.apply?.(this, arguments);
				let img;
				if (this.imageIndex != null) {
					// An image is selected so select that
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					// No image is selected but one is hovered
					img = this.imgs[this.overIndex];
				}
				if (img) {
					let pos = options.findIndex((o) => o.content === "Save Image");
					if (pos === -1) {
						pos = 0;
					} else {
						pos++;
					}
					options.splice(pos, 0, {
						content: "TilePrompt (McBoaty)",
						callback: async () => {
							let src = img.src;
							src = src.replace("/view?", `/MaraScott/McBoaty_v5/tile_prompt?node=${this.id}&clientId=${api.clientId}&`);
							const res = await (await fetch(src)).json();
							alert(res);
						},
					});
				}

				return r;
			};
		}
	},
});