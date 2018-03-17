import * as React from 'react';
import { Card } from 'antd';

export interface QuestionProps {
  question: string;
  type: 'option';
}

export interface QuestionState {
}

export default class Question extends React.Component<QuestionProps, QuestionState> {
  constructor(props: QuestionProps) {
    super(props);

  }

  render() {
    return (
      <Card />
    );
  }
}
