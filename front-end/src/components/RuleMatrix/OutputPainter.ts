import * as d3 from 'd3';
import * as nt from '../../service/num';
import { Painter, ColorType, labelColor, defaultDuration } from '../Painters';
import { RuleX } from './models';
import { registerStripePattern } from '../../service/utils';
import { isRuleGroup } from '../../models/ruleModel';

interface OptionalSupportParams {
  duration: number;
  color: ColorType;
}

interface SupportParams extends Partial<OptionalSupportParams> {
  widthFactor: number;
  height: number;
}

type SupportData = number[] | number[][];

function isMat(a: number[] | number[][]): a is number[][] {
  return Array.isArray(a[0]);
}

function registerPatterns(color: ColorType, keys: number[]) {
  return keys.map((key) => registerStripePattern(color(key), 3, 5));
}

export class SupportPainter implements Painter<SupportData, SupportParams> {
  public static defaultParams: OptionalSupportParams = {
    color: labelColor,
    duration: defaultDuration,
  };
  private params: SupportParams & OptionalSupportParams;
  private support: SupportData;
  update(params: SupportParams): this {
    this.params = {...(SupportPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  data(newData: SupportData): this {
    this.support = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
  ): this {
    const support = this.support;
    if (isMat(support)) {
      this.renderSimple(selector, []);
      this.renderMat(selector, support);
    } else {
      this.renderMat(selector, []);
      this.renderSimple(selector, support);
    }
    return this;
  }
  renderSimple<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
    support: number[]
  ): this {
    const {duration, height, widthFactor, color} = this.params;
    
    const xs = [0, ...(nt.cumsum(support))];

    // Render
    // Join
    const rects = selector.selectAll('rect.mo-support').data(support);
    // Enter
    const rectsEnter = rects.enter().append('rect').attr('class', 'mo-support')
      .attr('height', height);
    // Update
    const rectsUpdate = rectsEnter.merge(rects)
      .style('fill', (d, i) => color(i));
    // Transition
    rectsUpdate.transition().duration(duration)
      .attr('width', (d) => d * widthFactor)
      .attr('x', (d, i) => xs[i] * widthFactor + i * 1.5)
      .attr('height', height);
    // Exit
    rects.exit().transition().duration(duration)
      .attr('width', 1e-6).remove();
    return this;
  }

  renderMat<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
    support: number[][]
  ): this {
    const { height, widthFactor, duration, color } = this.params;
    const trueLabels = support.map((s: number[]) => nt.sum(s));
    // const total = nt.sum(trueLabels);
    // const width = total * widthFactor;
    const widths = trueLabels.map((l) => l * widthFactor);
    const xs = [0, ...(nt.cumsum(widths))];
    const ys = support.map((s, i) => s[i] / trueLabels[i] * height);
    // const heights = ys.map((y) => height - y);

    // Render True Rects
    const trueData = support
      .map((s, i) => ({width: widths[i], x: xs[i], height: ys[i], data: s, label: i}))
      .filter(v => v.width > 0);
    // Join
    const rects = selector.selectAll('rect.mo-support-true')
      .data(trueData);
    // Enter
    const rectsEnter = rects.enter().append('rect').attr('class', 'mo-support-true')
      .attr('height', height);
    // Update
    const rectsUpdate = rectsEnter.merge(rects)
      .style('fill', d => color(d.label))
      .style('stroke', d => color(d.label));
    // Transition
    rectsUpdate.transition().duration(duration)
      .attr('width', d => d.width)
      .attr('x', (d, i) => d.x + i * 1.5)
      .attr('height', d => d.height);
    // Exit
    rects.exit().transition().duration(duration)
      .attr('width', 1e-6).remove();

    // Register the stripes
    const stripeNames = registerPatterns(color, d3.range(trueLabels.length));
    
    // Render the misclassified part using stripes
    const root = selector.selectAll<SVGGElement, number[]>('g.mo-support-mat')
      .data(trueData);
    // enter
    const rootEnter = root.enter().append<SVGGElement>('g')
      .attr('class', 'mo-support-mat');

    // update
    const rootUpdate = rootEnter.merge(root).style('stroke', d => color(d.label));

    // update transition
    rootUpdate.transition().duration(duration)
      .attr('transform', (d, i) => `translate(${d.x + i * 1.5},${d.height})`);

    // root exit
    const exitTransition = root.exit().transition().duration(duration).remove();
    exitTransition.selectAll('rect.mo-support-mat').attr('width', 1e-6).attr('x', 1e-6);

    // stripe rects
    const stripeRects = rootUpdate.selectAll('rect.mo-support-mat')
    .data((d) => {
      // const xs = [0, ...(nt.cumsum(d))];
      const base = nt.sum(d.data) - d.data[d.label];
      let factor = base ? d.width / base : 0;
      const _widths = d.data.map((v, j) => j === d.label ? 0 : v * factor);
      const _xs = [0, ...nt.cumsum(_widths)];
      // console.log(factor); // tslint:disable-line
      const ret = d.data.map((v, j) => ({
        height: height - d.height, 
        width: _widths[j], x: _xs[j], label: j
      }));
      return ret.filter(r => r.width > 0);
    });
    const stripeEnter = stripeRects.enter().append('rect')
      .attr('class', 'mo-support-mat').attr('height', d => d.height);
    const stripeUpdate = stripeEnter.merge(stripeRects)
      // .classed('striped', d => d.striped)
      // .style('stroke', d => color(d.label))
      // .style('display', d => d.striped ? 'inline' : 'none')
      .style('fill', d => `url(#${stripeNames[d.label]})`);

    stripeUpdate.transition().duration(duration)
      .attr('height', d => d.height)
      .attr('width', d => d.width).attr('x', d => d.x);
    
    stripeRects.exit().transition().duration(duration)
      .attr('width', 1e-6).attr('x', 1e-6).remove();

    return this;
  }
}

interface OptionalParams {
  color: ColorType;
  duration: number;
  fontSize: number;
  widthFactor: number;
}

export interface OutputParams extends Partial<OptionalParams> {
  // feature2Idx: (feature: number) => number;
  onClick?: (feature: number, condition: number) => void;
}

export default class OutputPainter implements Painter<RuleX[], OutputParams> {
  public static defaultParams: OptionalParams = {
    color: labelColor,
    duration: defaultDuration,
    fontSize: 14,
    widthFactor: 200,
    // expandFactor: [4, 3],
  };
  private rules: RuleX[];
  private useMat: boolean;
  private params: OutputParams & OptionalParams;
  private supportPainter: SupportPainter;
  constructor() {
    this.params = {...(OutputPainter.defaultParams)};
    this.supportPainter = new SupportPainter();
  }
  
  update(params: OutputParams): this {
    this.params = {...(OutputPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  data(newData: RuleX[]): this {
    this.rules = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
  ): this {
    const { duration } = this.params;
    const rules = this.rules;
    this.useMat = rules.length > 0 && isMat(rules[0].support);

    const collapseYs = new Map<string, number>();
    rules.forEach((r) => isRuleGroup(r) && r.rules.forEach((_r) => collapseYs.set(`o-${_r.idx}`, r.y)));
    this.renderHeader(selector);
    // ROOT Group
    const groups = selector.selectAll<SVGGElement, {}>('g.matrix-outputs')
      .data<RuleX>(rules, function (r: RuleX) { return r ? `o-${r.idx}` : this.id; });
    // Enter
    const groupsEnter = groups.enter()
      .append<SVGGElement>('g')
      .attr('class', 'matrix-outputs')
      .attr('id', d => `o-${d.idx}`);
    // Update
    const groupsUpdate = groupsEnter.merge(groups)
      .classed('hidden', false).classed('visible', true);
    const updateTransition = groupsUpdate.transition().duration(duration)
      .attr('transform', d => `translate(10,${d.y})`);

    this.renderOutputs(groupsEnter, groupsUpdate, updateTransition);
    this.renderFidelity(groupsEnter, groupsUpdate, updateTransition);
    this.renderSupports(groupsEnter, groupsUpdate);

    // Exit
    groups.exit()
      .classed('hidden', true).classed('visible', false)
      .transition().duration(duration)
      .attr('transform', (d, i, nodes) => 
        `translate(5,${collapseYs.get(nodes[i].id)})`);
    return this;
  }

  public renderHeader(root: d3.Selection<SVGGElement, any, d3.BaseType, any>): this {
    // make sure the group exists
    // console.log('here'); // tslint:disable-line
    const {duration} = this.params;
    const rules = this.rules;

    // const confidence = nt.sum(rules.map((r) => r.totalSupport * r.output[r.label])) / totalSupport;
    root.selectAll('g.mo-headers').data(['g']).enter()
      .append('g').attr('class', 'mo-headers').attr('transform', 'translate(0,-20)');
    
    let headerTexts = ['Output (Pr)', 'Evidence'];
    let headerXs = [15, 80];
    let fillRatios = [0, 0];
    let rectWidths = [80, 67];

    if (this.useMat) {
      const totalSupport = nt.sum(rules.map((r) => r.totalSupport));
      const fidelity = nt.sum(
        rules.map(r => isMat(r.support) ? nt.sum(r.support.map(s => s[r.label])) : 0)
      ) / totalSupport;
    
      const acc = nt.sum(
        rules.map(r => isMat(r.support) ? nt.sum(r.support.map((s, i) => s[i])) : 0)
      ) / totalSupport;

      headerTexts = ['Output (Pr)', `Fidelity (${(fidelity * 100).toFixed(0)}/100)`, 
        `Evidence (${(acc * 100).toFixed(0)}/100)`];
      headerXs = [15, 75, 125];
      rectWidths = [80, 115, 125];
      fillRatios = [0, fidelity, acc];
    }

    const headers = root.select('g.mo-headers');

    const header = headers.selectAll('g.mo-header').data(headerTexts);
    const headerEnter = header.enter().append('g').attr('class', 'mo-header')
      .attr('transform', (d, i) => `translate(${headerXs[i]},0) rotate(-50)`)
      .style('font-size', 14);

    // rects
    headerEnter.append('rect')
      .attr('class', 'mo-header-box')
      .style('stroke-width', 1).style('stroke', '#1890ff').style('fill', '#fff')
      .attr('width', (d, i) => rectWidths[i]).attr('height', 20)
      .attr('rx', 2).attr('ry', 2);

    // rects
    headerEnter.append('rect').attr('class', 'mo-header-fill')
      .style('stroke-width', 1).style('stroke', '#1890ff')
      .style('fill', '#1890ff').style('fill-opacity', 0.1)
      .attr('height', 20)
      .attr('rx', 2).attr('ry', 2);

    // texts
    headerEnter.append('text')
      .attr('class', 'mo-header-text')
      .attr('text-anchor', 'start')
      .attr('fill', '#1890ff')
      .attr('dx', 3).attr('dy', 15);
    // Update
    const headerUpdate = headerEnter.merge(header);
    headerUpdate.select('text.mo-header-text').text(d => d);
    const transition = headerUpdate.transition().duration(duration)
      .attr('transform', (d, i) => `translate(${headerXs[i]},0) rotate(-50)`);
    transition.select('rect.mo-header-box').attr('width', (d, i) => rectWidths[i]);
    transition.select('rect.mo-header-fill')
      .attr('width', (d, i) => fillRatios[i] * rectWidths[i]);
    
    // textsEnter.merge(texts).text(d => d);
    return this;
  }

  public renderOutputs(
    enter: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    update: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    updateTransition: d3.Transition<SVGGElement, RuleX, SVGGElement, RuleX[]>
  ): this {
    const {fontSize, color, duration} = this.params;
    // const outputWidth = fontSize * 2;

    // *Output Texts*
    // Enter
    enter.append('text').attr('class', 'mo-output').attr('text-anchor', 'middle');
    // Update
    update.select('text.mo-output')
      .attr('font-size', d => isRuleGroup(d) ? fontSize * 0.8 : fontSize)
      .attr('dy', d => d.height / 2 + fontSize * 0.4)
      .attr('dx', 15)
      .text((d: RuleX) =>
        isRuleGroup(d) ? '' : (Math.round(d.output[d.label] * 100) / 100).toFixed(2)
      );  // confidence as text
    // Transition
    updateTransition.select('text.mo-output')
      .style('fill', d => 
        color(d.label)
        // d3.interpolateRgb.gamma(2.2)('#ccc', '#000')(d.output[d.label] * 2 - 1)
        // d3.interpolateRgb.gamma(2.2)('#ddd', color(d.label))(d.output[d.label] * 2 - 1)
      );

    // *Output Bars*
    const rectHeight = fontSize;
    enter.append('g').attr('class', 'mo-outputs');
    const rects = update.select('g.mo-outputs')
      .attr('transform', d => `translate(30,${d.height / 2 - fontSize * 0.4})`)
      .selectAll('rect')
      .data(d => {
        if (isRuleGroup(d)) return [];
        let y = 0;
        return d.output.map(o => {
          const ret = {o, y};
          y += o * rectHeight;
          return ret;
        });
      });
    
    const rectsUpdate = rects.enter().append('rect')
      .merge(rects);
    rectsUpdate.attr('width', 3).style('fill', (d, i) => color(i))
      .transition().duration(duration)
      .attr('height', d => d.o * rectHeight)
      .attr('y', d => d.y);
    
    rects.exit().transition().duration(duration)
      .style('fill-opacity', 1e-6).remove();

    enter.append('path').attr('class', 'mo-divider')
      .attr('stroke-width', 0.5)
      .attr('stroke', '#444')
      .attr('d', d => `M 60 0 V ${d.height}`);

    update.select('path.mo-divider').attr('d', d => `M 50 0 V ${d.height}`);
    return this;
  }

  public renderFidelity(
    enter: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    update: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    updateTransition: d3.Transition<SVGGElement, RuleX, SVGGElement, RuleX[]>
  ): this {
    const {fontSize, duration} = this.params;
    // const outputWidth = fontSize * 2;
    const dx = 80;
    const arc = d3.arc<any>().innerRadius(fontSize * 0.9).outerRadius(fontSize * 0.9 + 2).startAngle(0);
    // *Output Texts*
    // Enter
    const enterGroup = enter.append('g').attr('class', 'mo-fidelity');
    enterGroup.append('text').attr('class', 'mo-fidelity').attr('text-anchor', 'middle');
    enterGroup.append('path').attr('class', 'mo-fidelity').attr('d', arc({endAngle: 1e-6}) as string);

    // Update
    const updateGroup = update.select<SVGGElement>('g.mo-fidelity')
      .datum(d => {
        const fidelity = isMat(d.support) ? (nt.sum(d.support.map(s => s[d.label])) / d.totalSupport) : undefined;
        const color = fidelity ? (fidelity > 0.8 ? '#52c41a' :  fidelity > 0.5 ? '#faad14' : '#f5222d') : null;
        return {...d, fidelity, color};
      });
    updateGroup.select('text.mo-fidelity')
      .attr('font-size', d => isRuleGroup(d) ? fontSize * 0.8 : fontSize)
      .attr('dy', d => d.height / 2 + fontSize * 0.4)
      .attr('dx', dx)
      .text(d =>
        (!isRuleGroup(d) && d.fidelity) ? (Math.round(d.fidelity * 100)).toFixed(0) : ''
      )
      .style('fill', d => d.color);

    // Join
    updateGroup.transition().duration(duration).select('path.mo-fidelity')
      .attr('transform', d => `translate(${dx}, ${d.height / 2})`) // update pos
      .attr('d', d => arc({endAngle: (!isRuleGroup(d) && d.fidelity) ? (Math.PI * d.fidelity * 2) : 1e-6}))
      .style('fill', d => d.color);
    // Enter + Merge
    // const pathsUpdate = paths.enter()
    //   .append('path').attr('d', d => arc({endAngle: 0}))
    //   .attr('class', 'mo-fidelity')
    //   .merge(paths);

    // // transition
    // pathsUpdate.transition().duration(duration)
    //   .attr('d', d => arc({endAngle: Math.PI * d * 2}));
    
    // paths.exit().transition().duration(duration)
    //   .style('fill-opacity', 1e-6).remove();
    return this;
  }

  renderSupports(
    enter: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    update: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>
  ): this {
    const { duration, fontSize, widthFactor, color } = this.params;
    const useMat = this.useMat;
    // Enter
    enter.append('g').attr('class', 'mo-supports');
    // Update
    const supports = update.select<SVGGElement>('g.mo-supports');
    supports.transition().duration(duration)
      .attr('transform', `translate(${useMat ? (fontSize * 8) : (fontSize * 5)},0)`);

    // supports
    supports.each(({support, height}, i, nodes) => 
      support && this.supportPainter
        .update({widthFactor, height, color})
        .data(support)
        .render(d3.select(nodes[i]))
    );

    return this;
  }
}
