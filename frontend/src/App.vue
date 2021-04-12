<template>
  <v-app>
    <v-main>
      <v-container><v-row>
        <v-col cols="12" md="8" offset-md="2">
          <MyParaTTS/>
        </v-col>
      </v-row></v-container>
    </v-main>
  </v-app>
</template>

<script>
import MyParaTTS from './components/MyParaTTS'

export default {
  name: 'App',

  components: {
    MyParaTTS
  },

  data: () => ({
    //
  })
}
</script>
