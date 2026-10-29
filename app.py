"""Chinese desktop GUI. No network service and no sample upload."""
from pathlib import Path
import argparse
import csv
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from worker import ROOT as ENGINE_ROOT, read_input, write_csv, DOMAINS


def open_file(path):
    path = str(Path(path).resolve())
    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def value(x):
    if x is None or x == '':
        return '—'
    if isinstance(x, float):
        return f'{x:.4f}'
    return str(x)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('EDC-AOPRisk · 内分泌活性筛选')
        self.geometry('1240x820'); self.minsize(1020, 720)
        self.configure(bg='#f3f5f8')
        families = set(tkfont.families(self))
        self.ui_font = next((name for name in ['Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'] if name in families), 'Arial')
        self.option_add('*Font', (self.ui_font, 10))
        style = ttk.Style(self); style.theme_use('clam')
        style.configure('.', background='#f3f5f8', foreground='#263447', font=(self.ui_font, 10))
        style.configure('TButton', padding=(12, 7), background='#e6ecf2', borderwidth=0)
        style.map('TButton', background=[('active', '#dce5ee')])
        style.configure('Primary.TButton', background='#146f78', foreground='white', padding=(18, 9))
        style.map('Primary.TButton', background=[('disabled', '#a7b9bd'), ('active', '#0d5c64')])
        style.configure('TEntry', fieldbackground='white', padding=5)
        style.configure('TNotebook', borderwidth=0)
        style.configure('TNotebook.Tab', padding=(20, 10), background='#e7ecf2')
        style.map('TNotebook.Tab', background=[('selected', 'white')])
        style.configure('Treeview', rowheight=31, background='white', fieldbackground='white', borderwidth=0)
        style.configure('Treeview.Heading', padding=8, background='#e9eef3', font=(self.ui_font, 10, 'bold'))
        style.map('Treeview', background=[('selected', '#d5ebed')], foreground=[('selected', '#17434a')])
        self.process = None; self.messages = queue.Queue(); self.result = None; self.current_output = None
        self.started = 0; self.file_source = None
        self.status = tk.StringVar(value='就绪 · 所有预测在本机运行')
        self.out_base = tk.StringVar(value=str((Path.home() / 'Documents' if (Path.home() / 'Documents').exists() else Path.home()) / 'EDC_AOPRisk_Results'))
        self.smiles_col = tk.StringVar(value='SMILES'); self.name_col = tk.StringVar(value='Name'); self.cas_col = tk.StringVar(value='CAS')
        self.search = tk.StringVar(); self.only_hits = tk.BooleanVar(value=False)
        self.group = tk.StringVar(value='全部组'); self.scope = tk.StringVar(value='核心17（EATS）')
        self.notice = tk.StringVar(value='运行后，在这里查看全部化合物的筛选结果。')
        self._build()
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(200, self.poll)

    def _build(self):
        header = tk.Frame(self, bg='#173b4b', padx=25, pady=17); header.pack(fill='x')
        tk.Label(header, text='EDC-AOPRisk', bg='#173b4b', fg='white', font=('Arial', 23, 'bold')).pack(side='left')
        tk.Label(header, text='内分泌活性筛选工作台', bg='#173b4b', fg='#d5e9ec', font=(self.ui_font, 13)).pack(side='left', padx=22)
        tk.Label(header, text='V10B.1  |  GUI 1.0.2', bg='#173b4b', fg='#b3cbd3', font=('Arial', 10)).pack(side='right')
        body = ttk.Frame(self, padding=(22, 16)); body.pack(fill='both', expand=True)
        ttk.Label(body, textvariable=self.status, foreground='#486477', padding=(0, 10, 0, 0)).pack(side='bottom', anchor='w')
        self.tabs = ttk.Notebook(body); self.tabs.pack(fill='both', expand=True)
        self.input_tab = ttk.Frame(self.tabs, padding=20); self.summary_tab = ttk.Frame(self.tabs, padding=16)
        self.detail_tab = ttk.Frame(self.tabs, padding=16); self.info_tab = ttk.Frame(self.tabs, padding=16)
        for frame, name in [(self.input_tab, '1  导入与运行'), (self.summary_tab, '2  化合物汇总'),
                            (self.detail_tab, '3  终点明细'), (self.info_tab, '4  模型与规则')]:
            self.tabs.add(frame, text=name)
        ttk.Label(self.input_tab, text='导入化合物结构，筛选值得后续验证的内分泌活性线索', font=(self.ui_font, 15, 'bold')).pack(anchor='w')
        ttk.Label(self.input_tab, text='输入需要SMILES结构；名称和CAS用于标识。支持CSV、TSV或逐行粘贴SMILES。', foreground='#5c6b7b').pack(anchor='w', pady=(6, 14))
        top = ttk.Frame(self.input_tab); top.pack(fill='x')
        self.import_button = ttk.Button(top, text='选择 CSV / TSV', command=self.choose_input); self.import_button.pack(side='left')
        self.demo_button = ttk.Button(top, text='载入示例', command=self.load_demo); self.demo_button.pack(side='left', padx=8)
        self.clear_button = ttk.Button(top, text='清空并粘贴', command=self.clear_input); self.clear_button.pack(side='left')
        self.source_label = ttk.Label(top, text='可直接粘贴结构，或载入文件后预览', foreground='#627183'); self.source_label.pack(side='left', padx=14)
        editor_frame = ttk.Frame(self.input_tab); editor_frame.pack(fill='both', expand=True, pady=(12, 10))
        editor_frame.rowconfigure(0, weight=1); editor_frame.columnconfigure(0, weight=1)
        self.editor = tk.Text(editor_frame, height=7, wrap='none', bg='white', fg='#25364a', relief='solid', bd=1,
                              highlightthickness=0, font=('Consolas' if sys.platform == 'win32' else self.ui_font, 11), padx=12, pady=10, undo=True)
        self.editor.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(editor_frame, command=self.editor.yview); scroll.grid(row=0, column=1, sticky='ns'); self.editor.configure(yscrollcommand=scroll.set)
        scroll_x = ttk.Scrollbar(editor_frame, orient='horizontal', command=self.editor.xview); scroll_x.grid(row=1, column=0, sticky='ew'); self.editor.configure(xscrollcommand=scroll_x.set)
        cols = ttk.Frame(self.input_tab); cols.pack(fill='x', pady=(0, 12))
        self.column_boxes = []
        for text, variable in [('SMILES列', self.smiles_col), ('名称列（可选）', self.name_col), ('CAS列（可选）', self.cas_col)]:
            ttk.Label(cols, text=text).pack(side='left', padx=(0, 6))
            box = ttk.Combobox(cols, textvariable=variable, width=18); box.pack(side='left', padx=(0, 18)); self.column_boxes.append(box)
        directory = ttk.Frame(self.input_tab); directory.pack(fill='x')
        ttk.Label(directory, text='结果保存至').pack(side='left', padx=(0, 10))
        self.out_entry = ttk.Entry(directory, textvariable=self.out_base); self.out_entry.pack(side='left', fill='x', expand=True)
        self.dir_button = ttk.Button(directory, text='选择目录', command=self.choose_output); self.dir_button.pack(side='left', padx=(8, 0))
        buttons = ttk.Frame(self.input_tab); buttons.pack(fill='x', pady=(18, 10))
        self.run_button = ttk.Button(buttons, text='运行预测与筛选', style='Primary.TButton', command=self.start); self.run_button.pack(side='left')
        self.cancel_button = ttk.Button(buttons, text='停止', command=self.cancel, state='disabled'); self.cancel_button.pack(side='left', padx=8)
        self.progress = ttk.Progressbar(buttons, mode='indeterminate', length=200); self.progress.pack(side='left', padx=15)
        ttk.Label(self.input_tab, text='默认规则：全部17个EATS终点；数据库pAC50≥5.5，或符合论文门槛的域内模型候选。\n未入选不等于无活性或无风险。预测效力和活性概率分别呈现。', foreground='#617184', justify='left').pack(anchor='w', pady=(3, 0))
        bar = ttk.Frame(self.summary_tab); bar.pack(fill='x', pady=(0, 12))
        ttk.Label(bar, textvariable=self.notice, font=(self.ui_font, 11, 'bold')).pack(side='left')
        ttk.Button(bar, text='打开结果目录', command=self.open_output).pack(side='right')
        ttk.Button(bar, text='查看可打印报告', command=self.open_report).pack(side='right', padx=8)
        self.summary_tree = self.tree(self.summary_tab, ['输入序号', '化合物', '结果', '数据库入选终点数', '模型入选终点数', '关注终点'], [70, 220, 210, 130, 120, 420])
        self.summary_tree.bind('<<TreeviewSelect>>', self.select_compound)
        tools = ttk.Frame(self.detail_tab); tools.pack(fill='x', pady=(0, 10))
        ttk.Label(tools, text='查找名称 / 终点').pack(side='left')
        query = ttk.Entry(tools, textvariable=self.search, width=22); query.pack(side='left', padx=7)
        query.bind('<KeyRelease>', lambda e: self.refresh_detail())
        for variable, values, width in [(self.scope, ['核心17（EATS）', '辅助24（仅供机制支持）'], 24),
                                         (self.group, ['全部组', '雌激素', '雄激素', '甲状腺相关', '类固醇生成/转化'], 20)]:
            c = ttk.Combobox(tools, textvariable=variable, values=values, width=width, state='readonly'); c.pack(side='left', padx=5)
            c.bind('<<ComboboxSelected>>', lambda e: self.refresh_detail())
        ttk.Checkbutton(tools, text='仅看入选', variable=self.only_hits, command=self.refresh_detail).pack(side='left', padx=7)
        ttk.Label(self.detail_tab, text='概率表示模型阳性倾向；实测与预测pAC50分列。双击查看预测区间和模型用途。辅助24终点不参与筛选。', foreground='#617184').pack(anchor='w', pady=(0, 9))
        self.detail_cols = ['输入序号', '化合物', '终点', '筛选结果', '活性概率', '数据库pAC50_严格质控', '预测pAC50', '适用域']
        self.detail_tree = self.tree(self.detail_tab, self.detail_cols, [60, 150, 220, 115, 115, 140, 125, 165])
        self.detail_tree.bind('<Double-1>', self.show_detail)
        self.detail_count = tk.StringVar(value='尚无结果'); ttk.Label(self.detail_tab, textvariable=self.detail_count).pack(anchor='w', pady=(8, 0))
        info = '''这套模型可以做什么
• 输入化合物SMILES，调用已冻结的41终点活动模型、40个效力模型及63个阈值模型。
• 按论文规则，从全部17个EATS终点中筛选数据库证据和模型候选；辅助24终点另列。
• 所有计算在本机完成。安装依赖时需要联网；预测过程不查询外部服务。

默认筛选规则
数据库路径：至少一个EATS终点具有严格质控数据库pAC50≥5.5。
模型路径：该终点无严格/较宽质控实测效力，活性概率≥0.65，联合适用域内，
报告ROC-AUC≥0.70，PR-AUC高于训练阳性比例，且终点未被暂停筛选。
ESR2拮抗、TSHR/TRHR激动及两个定向类固醇生成终点不用于提升模型候选。

如何解释
数据库匹配针对标准化母体结构；输入盐类时会取主要片段并中和，请核对输出结构。
活性概率表示模型对特定体外终点的预测；不能作为人体患病概率或跨终点风险分值。
预测pAC50和区间保留供核查，弱效力模型不应被当作准确的定量毒性测定。
模型候选用于后续确证、迁移及生物活性验证。未入选不代表无活性或无风险。

版本与导出
模型：V10B.1 Audit-Frozen（2026-08-14），权重与冻结预测脚本逐字节保留。
筛选层：论文2026-10-28版本；ROC-AUC等采用补充表S8的报告协议数值。
每次运行单独保存中文CSV、可打印HTML报告、输入检查、运行环境及原始41终点输出。
双击“首次安装_Windows.bat”配置环境，然后双击“启动_Windows.bat”打开本界面。
'''
        text = tk.Text(self.info_tab, wrap='word', relief='flat', bg='white', padx=18, pady=14, font=(self.ui_font, 11))
        text.pack(fill='both', expand=True); text.insert('1.0', info); text.configure(state='disabled')

    def tree(self, parent, columns, widths):
        container = ttk.Frame(parent); container.pack(fill='both', expand=True)
        container.columnconfigure(0, weight=1); container.rowconfigure(0, weight=1)
        tree = ttk.Treeview(container, columns=columns, show='headings', selectmode='browse')
        for c, w in zip(columns, widths):
            tree.heading(c, text={'活性概率':'模型活性概率','数据库pAC50_严格质控':'数据库实测pAC50','预测pAC50':'模型预测pAC50'}.get(c,c)); tree.column(c, width=w, minwidth=60, stretch=False)
        tree.grid(row=0, column=0, sticky='nsew')
        y = ttk.Scrollbar(container, orient='vertical', command=tree.yview); y.grid(row=0, column=1, sticky='ns')
        x = ttk.Scrollbar(container, orient='horizontal', command=tree.xview); x.grid(row=1, column=0, sticky='ew')
        tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        tree.tag_configure('database', foreground='#09616d'); tree.tag_configure('model', foreground='#825814')
        tree.tag_configure('invalid', foreground='#ad3e40')
        return tree

    def choose_input(self):
        file = filedialog.askopenfilename(filetypes=[('化合物表格', '*.csv *.tsv'), ('全部文件', '*.*')])
        if file: self.load_file(Path(file))

    def load_file(self, path):
        data = path.read_bytes()
        encodings = ['utf-16'] if data.startswith((b'\xff\xfe', b'\xfe\xff')) else ['utf-8-sig', 'gb18030']
        for encoding in encodings:
            try: text = data.decode(encoding); break
            except UnicodeError: continue
        else: return messagebox.showerror('读取失败', '请另存为UTF-8 CSV。')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        if len(data) > 8_000_000:
            return messagebox.showerror('文件过大', '请分成不超过2000行的文件。')
        self.file_source = path
        self.editor.delete('1.0', 'end'); self.editor.insert('1.0', text)
        self.source_label.configure(text=path.name + '（可在此编辑；运行时使用当前文本）')
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        columns = next(csv.reader(text.splitlines(), delimiter=delimiter), [])
        for box in self.column_boxes: box.configure(values=columns)
        aliases = [(self.smiles_col, ['SMILES', 'smiles', 'Smiles', '结构']),
                   (self.name_col, ['Name', 'name', 'Chemical', '化合物', '名称']),
                   (self.cas_col, ['CAS', 'cas', 'CAS号'])]
        for variable, names in aliases:
            variable.set(next((n for n in names if n in columns), names[0]))

    def load_demo(self):
        self.load_file(ROOT / 'examples/demo_compounds.csv')

    def clear_input(self):
        self.editor.delete('1.0', 'end'); self.file_source = None
        self.source_label.configure(text='每行粘贴一个SMILES，或粘贴带表头的CSV')
        self.smiles_col.set('SMILES'); self.name_col.set('Name'); self.cas_col.set('CAS')

    def choose_output(self):
        path = filedialog.askdirectory(initialdir=str(Path.home()))
        if path: self.out_base.set(path)

    def set_busy(self, busy):
        self.run_button.configure(state='disabled' if busy else 'normal')
        for button in [self.import_button, self.demo_button, self.clear_button, self.dir_button, self.out_entry] + self.column_boxes:
            button.configure(state='disabled' if busy else 'normal')
        self.editor.configure(state='disabled' if busy else 'normal')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        if busy: self.progress.start(15)
        else: self.progress.stop()

    def start(self):
        if self.process: return
        text = self.editor.get('1.0', 'end').strip()
        if not text: return messagebox.showerror('没有输入', '请载入文件或粘贴SMILES。')
        try:
            base = Path(self.out_base.get()).expanduser().resolve(); base.mkdir(parents=True, exist_ok=True)
            run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            request_dir = base / ('input_' + run_id); request_dir.mkdir()
            delim = '\t' if (self.file_source and self.file_source.suffix.lower() == '.tsv') else ','
            header = next(csv.reader(text.splitlines(), delimiter=delim), [])
            if self.smiles_col.get() in header:
                inp = request_dir / ('compounds.tsv' if delim == '\t' else 'compounds.csv')
                inp.write_text(text + '\n', encoding='utf-8')
            else:
                if ',' in text.splitlines()[0] or '\t' in text.splitlines()[0]:
                    raise ValueError('请在“SMILES列”中选取文件的结构列名。')
                inp = request_dir / 'compounds.csv'
                write_csv(inp, [{'SMILES': s.strip()} for s in text.splitlines() if s.strip()])
                self.smiles_col.set('SMILES')
            read_input(inp, self.smiles_col.get(), self.name_col.get(), self.cas_col.get())
            out = base / ('result_' + run_id)
            request = {'input': str(inp), 'output': str(out), 'smiles_col': self.smiles_col.get(),
                       'name_col': self.name_col.get(), 'cas_col': self.cas_col.get()}
            request_file = request_dir / 'request.json'; request_file.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {'start_new_session': True}
            env = os.environ.copy(); env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1')
            self.process = subprocess.Popen([sys.executable, str(ROOT / 'worker.py'), '--request', str(request_file)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env, **options)
            self.result = None
            self.summary_tree.delete(*self.summary_tree.get_children())
            self.detail_tree.delete(*self.detail_tree.get_children())
            self.notice.set('运行中，完成后显示本次结果。')
            self.current_output = out; self.started = time.time(); self.status.set('启动预测器…'); self.set_busy(True)
            threading.Thread(target=self.read_process, args=(self.process,), daemon=True).start()
        except Exception as e:
            messagebox.showerror('无法运行', str(e))

    def read_process(self, process):
        lines = []
        for line in process.stdout:
            lines.append(line)
            try: self.messages.put(('progress', json.loads(line)))
            except (ValueError, TypeError): pass
        self.messages.put(('exit', {'code': process.wait(), 'log': ''.join(lines)}))

    def poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == 'progress':
                    self.status.set(payload.get('message', '') + f'  · 已用时 {int(time.time() - self.started)} 秒')
                elif kind == 'exit':
                    self.process = None; self.set_busy(False)
                    if self.current_output and self.current_output.exists():
                        (self.current_output / '界面运行日志.txt').write_text(payload['log'], encoding='utf-8')
                    if payload['code'] == 0:
                        self.load_results(self.current_output)
                        self.status.set('完成 · 结果已保存至 ' + str(self.current_output))
                    else:
                        self.status.set('本次运行未完成；已有输出仅供排查，不作为完整结果。')
                        if payload['code'] not in (-15, -9, 1 if getattr(self, 'was_cancelled', False) else -999):
                            tail = payload['log'].strip().splitlines()[-8:]
                            messagebox.showerror('预测未完成', '\n'.join(tail) or '请查看运行日志。')
                    self.was_cancelled = False
        except queue.Empty:
            pass
        self.after(200, self.poll)

    def load_results(self, output):
        result_file = Path(output) / 'results.json'
        result = json.loads(result_file.read_text(encoding='utf-8'))
        if result.get('status') != 'completed': raise ValueError('结果未完成')
        self.result = result; self.current_output = Path(output)
        self.notice.set(f"{result['n_input']}个输入 · {result['n_priority_compounds']}个关注候选 · {result['n_invalid']}个无效结构")
        self.summary_tree.delete(*self.summary_tree.get_children())
        columns = self.summary_tree['columns']
        for i, row in enumerate(result['summary']):
            tag = 'invalid' if row['结果'] == '输入无效' else 'database' if row['数据库入选终点数'] else 'model' if row['模型入选终点数'] else ''
            self.summary_tree.insert('', 'end', iid=str(i), values=[value(row[c]) for c in columns], tags=(tag,))
        self.search.set(''); self.only_hits.set(False); self.scope.set('核心17（EATS）'); self.refresh_detail()
        self.tabs.select(self.summary_tab)

    def select_compound(self, event=None):
        chosen = self.summary_tree.selection()
        if chosen and self.result:
            self.search.set(self.result['summary'][int(chosen[0])]['化合物'])
            self.refresh_detail()

    def refresh_detail(self):
        if not self.result: return
        rows = self.result['details']
        if self.scope.get().startswith('辅助'):
            file = self.current_output / '原始模型输出/endpoint_level_support24_v10b_external_gated.csv'
            raw = []
            if file.exists():
                with file.open(encoding='utf-8-sig') as f: raw = list(csv.DictReader(f))
            rows = [{'输入序号': int(r['input_row_id']), '化合物': r['Name'], 'EATS组': '辅助机制', '终点': r['endpoint_key'],
                     '筛选结果': '辅助，不参与筛选', '活性概率': float(r['activity_probability_calibrated_v10']),
                     '数据库pAC50_严格质控': r['observed_parent_pAC50_strict_v10'], '预测pAC50': r['predicted_pAC50_v10'],
                     '适用域': DOMAINS.get(r['joint_AD_v10'], r['joint_AD_v10']), **{'原始字段': r}} for r in raw] if file.exists() else []
        query = self.search.get().strip().lower()
        rows = [r for r in rows if (not query or query in (r['化合物'] + ' ' + r['终点']).lower())
                and (self.group.get() == '全部组' or r['EATS组'] == self.group.get())
                and (not self.only_hits.get() or r['筛选结果'] in ('数据库证据', '模型候选'))]
        self.visible_details = rows
        self.detail_tree.delete(*self.detail_tree.get_children())
        for i, row in enumerate(rows):
            self.detail_tree.insert('', 'end', iid=str(i), values=[value(row.get(c)) for c in self.detail_cols],
                                    tags=('database' if row['筛选结果'] == '数据库证据' else 'model' if row['筛选结果'] == '模型候选' else '',))
        self.detail_count.set(f'当前显示 {len(rows)} 个化合物–终点配对')

    def show_detail(self, event=None):
        chosen = self.detail_tree.selection()
        if not chosen: return
        row = self.visible_details[int(chosen[0])]
        window = tk.Toplevel(self); window.title(row['化合物'] + ' · ' + row['终点']); window.geometry('850x650')
        text = tk.Text(window, wrap='word', bg='white', padx=22, pady=18); text.pack(fill='both', expand=True)
        for k, v in row.items(): text.insert('end', k + '：' + (json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, dict) else value(v)) + '\n\n')
        text.configure(state='disabled')

    def open_output(self):
        if self.current_output and self.current_output.exists(): open_file(self.current_output)

    def open_report(self):
        if self.current_output and (self.current_output / '结果报告.html').is_file(): open_file(self.current_output / '结果报告.html')

    def cancel(self):
        if not self.process: return
        self.was_cancelled = True
        if sys.platform == 'win32':
            subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try: os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        self.status.set('正在停止，未完成的结果不会作为完整预测显示。')

    def close(self):
        if self.process: self.cancel()
        self.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--open-results', type=Path)
    args = parser.parse_args()
    app = Application()
    if args.open_results: app.load_results(args.open_results)
    app.mainloop()
