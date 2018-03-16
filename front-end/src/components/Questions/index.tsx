import * as React from 'react';
import { Affix, Card } from 'antd';

export interface QuestionsProps {
}

export interface QuestionsState {
}

export default class Questions extends React.Component<QuestionsProps, QuestionsState> {
  constructor(props: QuestionsProps) {
    super(props);

  }

  render() {
    return (
      <div>
        <Affix>
          <Card />
        </Affix>
      </div>
    );
  }
}
