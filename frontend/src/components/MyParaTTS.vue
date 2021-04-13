<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-card><v-card-text>
          <v-container><v-row><v-col>
            <v-textarea
              v-model="text"
              counter
              solo
              name="input-7-4"
              label="请输入英文文本"
              auto-grow
              rows="7"
            ></v-textarea>
          </v-col></v-row></v-container>

          <v-container><v-row><v-col>
            <v-slider
              v-model="speed"
              max="5" min="1" step="1"
              thumb-label
              thumb-size="24"
              ticks
              label="语速"
              style="height: 30px"
              class="align-center"
            ></v-slider>

            <v-slider
              v-model="volume"
              max="5" min="1" step="1"
              thumb-label
              thumb-size="24"
              ticks
              label="音量"
              style="height: 30px"
              class="align-center"
            ></v-slider>

            <v-slider
              v-model="tone"
              max="5" min="1" step="1"
              thumb-label
              thumb-size="24"
              ticks
              label="语调"
              style="height: 30px"
              class="align-center"
            ></v-slider>

            <v-btn block elevation="2" color="primary" @click="getAudioFromBackend">合成</v-btn>
          </v-col></v-row></v-container>

          <v-container><v-row><v-col>
            <vue-audio-native
              ref="MyWave"
              :url="msource"
              :show-current-time="true"
              :show-controls="false"
              :show-download="true"
              :autoplay="true"
              :wait-buffer="true"/>
          </v-col></v-row></v-container>
        </v-card-text></v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from 'axios'

export default {
  data () {
    return {
      wave_url: 'demo.wav',
      speed: 4,
      volume: 3,
      tone: 3,
      text: 'Welcome to my parallel text to speech system !'
    }
  },
  methods: {
    // Ref: https://blog.csdn.net/qq_41844169/article/details/101028469
    getAudioFromBackend () {
      const path = '/api/mytts'
      axios({
        url: path,
        data: {
          speed: this.speed,
          volume: this.volume,
          tone: this.tone,
          text: this.text
        },
        method: 'post',
        responseType: 'blob'
      })
        .then(res => {
          this.$refs.MyWave.url = URL.createObjectURL(res.data)
        })
        .catch(error => {
          console.log(error)
        })
    }
  },
  created () {
    this.getAudioFromBackend()
  }
}
</script>
